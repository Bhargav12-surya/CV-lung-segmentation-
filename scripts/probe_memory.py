"""Phase C: Memory validation and preflight allocation probes across presets."""

import json
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lung_ct_segmentation import (
    ModelConfig,
    preflight_model,
    PRESETS,
    doctor_command,
    now,
    atomic_json
)

def run_memory_probes():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)
    if device.type != "cuda":
        print("CUDA device required for GPU VRAM measurement.", flush=True)
        return
        
    props = torch.cuda.get_device_properties(0)
    total_vram_gib = props.total_memory / (1024**3)
    gpu_name = props.name
    print(f"Device: {gpu_name} (Total VRAM: {total_vram_gib:.2f} GiB)\n", flush=True)
    
    presets_to_test = [
        {"name": "compact", "patch": (32, 96, 96), "widths": (16, 32, 64, 128, 192)},
        {"name": "medium", "patch": (48, 128, 128), "widths": (16, 32, 64, 128, 192)},
        {"name": "large_48x160", "patch": (48, 160, 160), "widths": (16, 32, 64, 128, 192)},
        {"name": "large_64x128", "patch": (64, 128, 128), "widths": (16, 32, 64, 128, 192)},
    ]
    
    probe_results = []
    
    for p in presets_to_test:
        patch = p["patch"]
        widths = p["widths"]
        name = p["name"]
        print(f"--- Probing Preset: '{name}' with Patch={patch}, Widths={widths} ---", flush=True)
        
        cfg = ModelConfig(
            name="hybrid3d",
            channels=2,
            widths=widths,
            early_anisotropic=True,
            multiscale=True,
            transformer=True,
            adaptive_fusion=True,
            wavelet=True,
            decoder_conditioning=True,
            acquisition_conditioning=True,
            boundary=True,
            deep_supervision=True,
            spatial_gate=False
        )
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        
        try:
            # Execute 2 full backward & optimizer steps via preflight_model
            res = preflight_model(
                config=cfg,
                patch=patch,
                batch_size=1,
                device=device,
                precision="bf16", # project standard BF16 precision
                lambda_ds=1.0,
                lambda_boundary=0.1,
                real_batch=None, # synthetic batch with positive tumor target
                loss_type="dice_bce"
            )
            
            peak_alloc_gib = torch.cuda.max_memory_allocated(0) / (1024**3)
            peak_res_gib = torch.cuda.max_memory_reserved(0) / (1024**3)
            
            probe_info = {
                "preset": name,
                "status": "PASSED",
                "patch": list(patch),
                "widths": list(widths),
                "parameters": res["parameters"],
                "precision": "bf16",
                "peak_allocated_gib": peak_alloc_gib,
                "peak_reserved_gib": peak_res_gib,
                "total_vram_gib": total_vram_gib,
                "free_headroom_gib": total_vram_gib - peak_res_gib,
                "gradient_norm": res["gradient_norm"],
                "step_seconds": res["seconds"] / res["optimizer_steps"]
            }
            print(f"  Result: PASSED!", flush=True)
            print(f"  Peak Allocated: {peak_alloc_gib:.3f} GiB | Peak Reserved: {peak_res_gib:.3f} GiB", flush=True)
            print(f"  Headroom: {total_vram_gib - peak_res_gib:.3f} GiB out of {total_vram_gib:.2f} GiB", flush=True)
            print(f"  Parameters: {res['parameters']:,} | Time/Step: {probe_info['step_seconds']:.3f}s\n", flush=True)
            probe_results.append(probe_info)
        except Exception as e:
            print(f"  Result: FAILED with error: {e}\n", flush=True)
            probe_results.append({
                "preset": name,
                "status": "FAILED",
                "patch": list(patch),
                "error": str(e)
            })
            
    out_json = REPO_ROOT / "evaluations" / "optimization_phase_c_preflight.json"
    atomic_json(out_json, {
        "date": now(),
        "gpu": gpu_name,
        "total_vram_gib": total_vram_gib,
        "probes": probe_results
    })
    print(f"Saved preflight probe report to: {out_json}", flush=True)

if __name__ == "__main__":
    run_memory_probes()
