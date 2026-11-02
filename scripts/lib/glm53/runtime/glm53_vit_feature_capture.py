"""Opt-in ViT output capture, disabled unless GLM53_VIT_CAPTURE_DIR is set."""
import functools
import hashlib
import logging
import os
from pathlib import Path

def install(cls, side):
    if not os.environ.get("GLM53_VIT_CAPTURE_DIR") or getattr(cls, "_glm53_capture_wrapped", False):
        return
    original = cls.forward
    seen = set()
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if len(seen) >= 16:
            return output
        import torch
        values = args[0] if args else kwargs.get("pixel_values", kwargs.get("hidden_states", kwargs.get("x")))
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        if not isinstance(features, torch.Tensor) or values is None or grid is None:
            raise RuntimeError("ViT capture requires tensor input, grid_thw and tensor/pooler_output")
        grids = grid.detach().cpu().tolist() if isinstance(grid, torch.Tensor) else grid
        merge = getattr(self, "spatial_merge_size", None)
        if merge is None:
            merge = self.config.spatial_merge_size
        counts = [int(t*h*w) for t,h,w in grids]
        outcounts = [n // (merge*merge) for n in counts]
        assert sum(counts) == values.shape[0], (counts, values.shape)
        assert sum(outcounts) == features.shape[0], (outcounts, features.shape)
        vi = fi = 0
        root = Path(os.environ["GLM53_VIT_CAPTURE_DIR"]) / side
        root.mkdir(parents=True, exist_ok=True)
        for g, n, m in zip(grids, counts, outcounts):
            pixels = values[vi:vi+n].detach().cpu().contiguous()
            key = hashlib.sha256(str(g).encode() + pixels.view(torch.uint8).numpy().tobytes()).hexdigest()
            if key not in seen and len(seen) < 16:
                path = root / (key + "-" + str(os.getpid()) + ".pt")
                torch.save({"input_hash":key, "grid":g, "input_shape":list(pixels.shape),
                            "input_dtype":str(pixels.dtype),
                            "features":features[fi:fi+m].detach().cpu().clone(),
                            "side":side, "pid":os.getpid()}, path)
                seen.add(key)
                logging.getLogger(__name__).info("VIT_FEATURE_CAPTURE side=%s input_hash=%s features=%d file=%s",side,key,m,path)
            vi += n
            fi += m
        return output
    cls.forward = forward
    cls._glm53_capture_wrapped = True
