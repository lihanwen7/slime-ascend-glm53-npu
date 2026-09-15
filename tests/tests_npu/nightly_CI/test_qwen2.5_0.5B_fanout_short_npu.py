"""E2E test: one prompt → random 1..3 training samples (compact / subagent fan-out).

What this test pins
-------------------
The "compact" pattern (where one rollout execution emits a *variable*
number of training samples sharing a single ``rollout_id``) has CPU unit
coverage at the piece-level (``test_dp_schedule.py`` for the rollout-
aware step splitter, ``test_sample.py`` for ``Sample.rollout_id`` round-
trip, ``test_cp_utils.py`` for the per-rollout-mean reducer). But until
this test, **no e2e training run had ever exercised the full chain**:

  custom_generate returns list[Sample] sharing rollout_id
    → _validate_rollout_id_annotated at depth ≥ 2 passes
    → _split_train_data_by_dp groups by rollout_id and trims to N steps
       using ``rollout_batch_size * n_samples_per_prompt / global_batch_size``
       (NOT total sample count, which would inflate steps once N>1)
    → loss aggregation uses ``rollout_mask_sums`` so every sibling sample
       contributes one token-weighted mean per rollout
    → train_one_step's ``step_global_batch_size`` denominator equals
       num_rollouts (not num_samples), keeping grad magnitude stable
       independent of fan-out

The fan-out function itself lives in
``slime/rollout/_fanout_test_helpers.py`` — it has to be at a dot-free
module path so ``importlib.import_module`` can resolve the string
``--custom-generate-function-path`` flag (this filename has dots).

Test choices
------------
- **Deterministic fan-out** ``N = 1 + (sample.index % MAX_FANOUT)`` for
  reproducibility. Every value in {1, 2, 3} gets exercised in a single
  rollout. N=1 keeps the backward-compat (no fan-out) path alive in CI.
- **Smoke + implicit step-count assertion**. ``--ci-test`` flips the
  framework's built-in numerical guards (KL divergence, log_prob ≈
  ref_log_prob); a step-counting / loss-denominator regression would
  trip them. Plus the helper writes one line per call to a tmp counter
  file — post-train we assert the count equals
  ``num_rollout * rollout_batch_size``, proving the custom path actually
  drove every prompt (vs. silent fallback to default rollout).
"""

import os
import tempfile

import slime.utils.external_utils.command_utils as U
HF_WEIGHTS_ROOT = os.environ.get(
    "SLIME_TEST_HF_WEIGHTS_ROOT",
    "/home/data/weights",
)

DATASETS_ROOT = os.environ.get(
    "SLIME_TEST_DATASETS_ROOT",
    "/home/data/datasets",
)

MEGATRON_WEIGHTS_ROOT = os.environ.get(
    "SLIME_TEST_MEGATRON_WEIGHTS_ROOT",
    "/home/c00880099/weights-MA",
)

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


# Counter file used by the compact_generate helper.
# We pass its path through to the Ray-submitted job via an env var so all
# worker processes write to the same path.
FANOUT_COUNTER_FILE = os.environ.get(
    "SLIME_FANOUT_TEST_COUNTER_FILE",
    os.path.join(
        tempfile.gettempdir(),
        "slime_fanout_test_counter.log",
    ),
)


def prepare():
    # Clear the counter so a previous run's invocations don't bleed in.
    try:
        os.remove(FANOUT_COUNTER_FILE)
    except FileNotFoundError:
        pass


def execute():
    ckpt_args = (
        f"--hf-checkpoint {HF_WEIGHTS_ROOT}/{MODEL_NAME}/ "
        f"--ref-load "
        f"{MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist "
    )

    # All fan-out is owned by compact_generate. This knob stays at 1 so a
    # regression that confuses sample count vs rollout count surfaces.
    # Four prompts retain deterministic fan-out values 1, 2 and 3 while
    # keeping global_batch_size divisible by four-way data parallelism.
    #
    # NB no --group-rm:
    #
    # When custom_generate returns list[Sample], the per-sample rm path
    # inside generate_and_rm handles the fan-out correctly via
    # batched_async_rm on the flat sibling list.
    #
    # --group-rm defers rm to generate_and_rm_group, which assumes group is
    # already flat list[Sample]. Combining it with a list-returning
    # custom_generate yields list[list[Sample]] and crashes async_rm with:
    #
    #   'list' object has no attribute 'metadata'
    #
    rollout_args = (
        f"--prompt-data "
        f"{DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 4 "
        "--balance-data "
        "--custom-generate-function-path "
        "slime.rollout._fanout_test_helpers.compact_generate "
        "--custom-reward-post-process-path "
        "slime.rollout._fanout_test_helpers."
        "grpo_normalize_by_group_index "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.4 "
        "--sglang-cuda-graph-max-bs 4 "
        "--sglang-enable-metrics "
        "--sglang-device npu "
    )

    ci_args = "--ci-test "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{fault_tolerance_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        # Make the counter path visible inside the Ray-submitted job.
        # The helper picks it up via os.environ.
        extra_env_vars={
            "SLIME_FANOUT_TEST_COUNTER_FILE": FANOUT_COUNTER_FILE,
        },
    )

    # Post-train assertion:
    #
    # compact_generate must have been called exactly
    #
    #   num_rollout * rollout_batch_size
    #   = 1 * 4
    #   = 4
    #
    # A regression that bypassed the custom path would either skip the
    # counter file entirely or under-count.
    expected_calls = 1 * 4

    try:
        with open(FANOUT_COUNTER_FILE) as f:
            actual_calls = sum(1 for _ in f)
    except FileNotFoundError as e:
        raise AssertionError(
            f"compact_generate counter file "
            f"{FANOUT_COUNTER_FILE} missing — the custom generate path "
            f"was never invoked. Suggests "
            f"--custom-generate-function-path was dropped by the arg "
            f"parser or the resolved import path is wrong."
        ) from e

    assert actual_calls == expected_calls, (
        f"compact_generate was called {actual_calls} times, "
        f"expected {expected_calls} "
        f"(num_rollout=1 x rollout_batch_size=4). "
        f"A mismatch points at the rollout submission loop "
        f"double-submitting / under-submitting prompts."
    )


if __name__ == "__main__":
    prepare()

    # Disable proxy safely. Using pop(..., None) avoids KeyError when a
    # proxy variable is not currently defined.
    for proxy_var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        os.environ.pop(proxy_var, None)

    execute()
