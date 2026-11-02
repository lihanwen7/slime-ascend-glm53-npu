"""Opt-in per-step allocator peaks; no operator profiling overhead."""
import os,json,time
from pathlib import Path
def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import torch
    import torch.distributed as dist
    root=os.environ.get("GLM53_MEMORY_METRICS_DIR")
    if not root: return
    rank=dist.get_rank()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    state={"rollout_id":rollout_id,"step_id":step_id,"rank":rank,"start":time.perf_counter()}
    optimizer._glm53_memory_state=state
    if getattr(optimizer,"_glm53_memory_wrapped",False): return
    original=optimizer.step
    def measured_step(*a,**kw):
        result=original(*a,**kw)
        torch.npu.synchronize()
        s=optimizer._glm53_memory_state
        row={k:v for k,v in s.items() if k!="start"}
        row.update(elapsed_s=time.perf_counter()-s["start"],
          peak_allocated_GiB=torch.npu.max_memory_allocated()/2**30,
          peak_reserved_GiB=torch.npu.max_memory_reserved()/2**30,
          allocated_GiB=torch.npu.memory_allocated()/2**30,
          reserved_GiB=torch.npu.memory_reserved()/2**30)
        Path(root).mkdir(parents=True,exist_ok=True)
        with open(Path(root)/f"rank{rank}.jsonl","a") as f:f.write(json.dumps(row)+"\n")
        print("GLM53_MEMORY_METRICS "+json.dumps(row),flush=True)
        return result
    optimizer.step=measured_step
    optimizer._glm53_memory_wrapped=True
