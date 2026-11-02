"""Optional NPU profiling of Megatron training steps (slime hook).

Loaded through ``--custom-megatron-before-train-step-hook-path``.  The hook is
called by ``slime/backends/megatron_utils/model.py::train_one_step`` before
every optimizer step on every training rank, so it profiles the training phase
only (forward / backward / optimizer); SGLang rollout runs in separate
processes and is never captured.

Steps are addressed by a per-process **global training-step counter** that
matches the ``train/step: N`` value printed in slime's logs (0-based).
When disabled (the default) the hook costs one ``os.environ.get`` per step.

Environment variables (normally injected via ``--train-env-vars``):

  GLM53_PROFILE_TRAIN=1          enable profiling
  GLM53_PROFILE_STEP_START=<int> first global train step to record (default 3)
  GLM53_PROFILE_STEP_END=<int>   last global train step to record (default 3)
  GLM53_PROFILE_DIR=<dir>        output root (default project cache/work/glm53-profiles)
  GLM53_PROFILE_RANKS=0,4        comma list of global ranks (empty = all)
  GLM53_PROFILE_LEVEL=Level1     Level0 / Level1 / Level2
  GLM53_PROFILE_AIC=AiCoreNone   AiCMetrics member (Level1/Level2 only)
  GLM53_PROFILE_WITH_STACK=0     1 = record python call stacks
  GLM53_PROFILE_MEMORY=0         1 = record profiler memory events
  GLM53_PROFILE_MICROBATCHES=0   >0 = profile only N micro-batches (+optimizer) of the step
  GLM53_MEM_SNAPSHOT=0           1 = record one step's allocator history
  GLM53_MEM_SNAPSHOT_STEP=<int>  step to record (default 1)
  GLM53_MEM_SNAPSHOT_FILE=<path> pickle output (default /tmp/glm53_mem_step<N>.pickle)
  GLM53_MEM_SNAPSHOT_MAX_ENTRIES=<int> history entries (default 1000000)

Output layout::

  $GLM53_PROFILE_DIR/trainstep<S>-<E>/rank<N>/
    <host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
    <host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
    <host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
    <host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv

``trace_view.json`` opens in Perfetto (https://ui.perfetto.dev) or
chrome://tracing: it contains the NPU kernel timeline, HCCL ops, and the
Python/CPU side with ``ProfilerStep#N`` markers.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Per-process state.  A training actor is reused across rollouts, so the
# profiler must survive between train_step calls and across rollouts.
_STATE = {"prof": None, "saved": False, "done": False, "rank": None, "step_calls": 0}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _global_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _selected(rank: int) -> bool:
    raw = os.environ.get("GLM53_PROFILE_RANKS", "").strip()
    if not raw:
        return True
    try:
        return rank in {int(item) for item in raw.split(",") if item.strip()}
    except ValueError:
        logger.warning("[glm53-profile] bad GLM53_PROFILE_RANKS=%r, profiling all ranks", raw)
        return True


def _start(rank: int, step_index: int, active: int) -> None:
    import torch_npu  # noqa: F401  registers the NPU profiler backend
    import torch_npu.profiler as npu_prof

    level_name = os.environ.get("GLM53_PROFILE_LEVEL", "Level1")
    aic_name = os.environ.get("GLM53_PROFILE_AIC", "AiCoreNone")
    level = getattr(npu_prof.ProfilerLevel, level_name, npu_prof.ProfilerLevel.Level1)
    aic = getattr(npu_prof.AiCMetrics, aic_name, npu_prof.AiCMetrics.AiCoreNone)
    microbatch_target = _env_int("GLM53_PROFILE_MICROBATCHES", 0)

    first = step_index
    last = step_index if microbatch_target > 0 else step_index + active - 1
    root = os.environ.get("GLM53_PROFILE_DIR", "/mnt/share/l00804615/glm5.3_flash_native/cache/work/glm53-profiles")
    out_dir = os.path.join(root, f"trainstep{first}-{last}", f"rank{rank}")
    os.makedirs(out_dir, exist_ok=True)

    def _on_ready(_prof) -> None:
        _STATE["saved"] = True
        logger.info("[glm53-profile] rank %s: trace saved under %s", rank, out_dir)

    prof = npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=npu_prof.schedule(
            wait=0, warmup=0,
            active=(microbatch_target + 1 if microbatch_target > 0 else active),
            repeat=1,
        ),
        on_trace_ready=npu_prof.tensorboard_trace_handler(out_dir, worker_name=f"rank{rank}"),
        record_shapes=False,
        profile_memory=os.environ.get("GLM53_PROFILE_MEMORY", "0") == "1",
        with_stack=os.environ.get("GLM53_PROFILE_WITH_STACK", "0") == "1",
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=level,
            aic_metrics=aic,
            data_simplification=True,
            export_type="text",
        ),
    )
    prof.start()

    _STATE.update(
        prof=prof, saved=False, done=False,
        micro_target=microbatch_target, micro_calls=0,
        micro_done=False, micro_start_step=step_index,
    )
    logger.info(
        "[glm53-profile] rank %s: profiling global train steps %s-%s -> %s",
        rank, first, last, out_dir,
    )


def on_microbatch() -> None:
    """Advance the profiler after one training forward step."""
    prof = _STATE.get("prof")
    if prof is None or _STATE.get("done") or _STATE.get("saved"):
        return
    target = _STATE.get("micro_target", 0)
    if target <= 0:
        return
    try:
        import torch

        torch.npu.synchronize()
    except Exception:
        pass
    prof.step()
    _STATE["micro_calls"] = _STATE.get("micro_calls", 0) + 1
    if _STATE.get("saved"):
        _STATE["micro_done"] = True


_MEM_STATE = {"recording": False}


def _mem_snapshot(rank: int, step_index: int) -> None:
    """Record one train step's allocator history and dump a pickle snapshot.

    The dump happens on the next train step's hook, so a train step must exist
    after ``GLM53_MEM_SNAPSHOT_STEP`` (e.g. num_rollout >= target + 2).
    """
    if os.environ.get("GLM53_MEM_SNAPSHOT", "0") != "1" or rank != 0:
        return
    import torch_npu
    import torch_npu.npu.memory as npu_mem

    target = _env_int("GLM53_MEM_SNAPSHOT_STEP", 1)
    if step_index == target:
        max_entries = _env_int("GLM53_MEM_SNAPSHOT_MAX_ENTRIES", 1_000_000)
        npu_mem._record_memory_history(
            enabled="all", stacks="all", max_entries=max_entries
        )
        _MEM_STATE["recording"] = True
        logger.info(
            "[glm53-mem] rank %s: recording memory history at step %s "
            "(stacks=all, max_entries=%s)",
            rank, step_index, max_entries,
        )
    elif _MEM_STATE["recording"]:
        path = os.environ.get(
            "GLM53_MEM_SNAPSHOT_FILE", f"/tmp/glm53_mem_step{target}.pickle"
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            npu_mem._dump_snapshot(path)
        finally:
            npu_mem._record_memory_history(enabled=None)
            _MEM_STATE["recording"] = False
        logger.info("[glm53-mem] rank %s: snapshot saved to %s", rank, path)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Hook entry point, called before every Megatron training step."""
    profiling = os.environ.get("GLM53_PROFILE_TRAIN", "0") == "1"
    mem_snapshot = os.environ.get("GLM53_MEM_SNAPSHOT", "0") == "1"
    if not profiling and not mem_snapshot:
        return

    try:
        rank = _STATE["rank"]
        if rank is None:
            rank = _global_rank()
            _STATE["rank"] = rank

        # Global training-step index; matches slime's ``train/step`` log value.
        step_index = _STATE["step_calls"]
        _STATE["step_calls"] = step_index + 1

        _mem_snapshot(rank, step_index)

        if not profiling:
            return
        if not _selected(rank):
            return

        start = _env_int("GLM53_PROFILE_STEP_START", 3)
        end = max(_env_int("GLM53_PROFILE_STEP_END", start), start)

        prof = _STATE["prof"]
        if prof is None:
            if not _STATE["done"] and start <= step_index <= end:
                _start(rank, step_index, end - step_index + 1)
            return

        micro_target = _STATE.get("micro_target", 0)
        if micro_target > 0:
            if _STATE.get("saved"):
                prof.stop()
                _STATE["prof"] = None
                _STATE["done"] = True
                return
            try:
                import torch

                torch.npu.synchronize()
            except Exception:
                pass
            for _ in range(max(1, micro_target + 1 - _STATE.get("micro_calls", 0))):
                prof.step()
                if _STATE["saved"]:
                    break
            _STATE["done"] = True
            if _STATE["saved"]:
                prof.stop()
                _STATE["prof"] = None
            return

        # Make the trace window match optimizer-step boundaries: flush pending
        # NPU work before advancing the profiler schedule.
        try:
            import torch

            torch.npu.synchronize()
        except Exception:
            pass

        prof.step()

        if _STATE["saved"]:
            prof.stop()
            _STATE["prof"] = None
            _STATE["done"] = True
    except Exception:
        logger.exception("[glm53-profile] hook failed; disabling profiling for this rank")
        try:
            prof = _STATE.get("prof")
            if prof is not None:
                prof.stop()
        except Exception:
            pass
        _STATE.update(prof=None, saved=True, done=True)
