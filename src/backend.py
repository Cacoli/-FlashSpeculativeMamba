"""Backend detection and hardware capability diagnostics for FlashSpeculativeMamba."""

import sys
import platform
from typing import Dict, Any

def get_backend_info() -> Dict[str, Any]:
    """Inspect and return current hardware, CUDA, and Mamba backend status."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "total_vram_gb": None,
        "free_vram_gb": None,
        "has_mamba_ssm": False,
        "has_causal_conv1d": False,
        "backend_name": "unknown",
        "is_slow_path": True,
    }

    # Inspect PyTorch and CUDA
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if hasattr(torch.version, "cuda") else None

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["total_vram_gb"] = round(total_bytes / (1024 ** 3), 2)
            info["free_vram_gb"] = round(free_bytes / (1024 ** 3), 2)
    except ImportError:
        pass

    # Inspect fused CUDA kernel packages
    try:
        import mamba_ssm  # noqa: F401
        info["has_mamba_ssm"] = True
    except ImportError:
        info["has_mamba_ssm"] = False

    try:
        import causal_conv1d  # noqa: F401
        info["has_causal_conv1d"] = True
    except ImportError:
        info["has_causal_conv1d"] = False

    if info["has_mamba_ssm"] and info["has_causal_conv1d"]:
        info["backend_name"] = "fused_mamba_cuda"
        info["is_slow_path"] = False
    else:
        info["backend_name"] = "hf_pure_pytorch"
        info["is_slow_path"] = True

    return info

def print_backend_summary() -> None:
    """Print human-readable backend and CUDA status to stdout."""
    info = get_backend_info()
    print("=" * 60)
    print("FlashSpeculativeMamba Backend & Hardware Diagnostics")
    print("=" * 60)
    print(f"Python Version           : {info['python_version']}")
    print(f"PyTorch Version          : {info['torch_version'] if info['torch_version'] else 'Not Installed'}")
    print(f"CUDA Available           : {info['cuda_available']}")
    print(f"CUDA Version (torch)     : {info['cuda_version'] if info['cuda_version'] else 'N/A'}")
    print(f"GPU Device Name          : {info['gpu_name'] if info['gpu_name'] else 'None / CPU'}")
    if info["total_vram_gb"] is not None:
        print(f"VRAM (Free / Total)      : {info['free_vram_gb']} GB / {info['total_vram_gb']} GB")
    else:
        print("VRAM                     : N/A")
    print(f"mamba-ssm installed      : {info['has_mamba_ssm']}")
    print(f"causal-conv1d installed  : {info['has_causal_conv1d']}")
    print(f"Active Backend           : {info['backend_name']}")
    print("-" * 60)

    if info["is_slow_path"]:
        print("[WARNING] Active backend is 'hf_pure_pytorch' (slow path).")
        print("          Fused CUDA kernels (mamba-ssm, causal-conv1d) are NOT installed.")
        print("          PyTorch eager mode is used for recurrent state stepping.")
        print("          Never compare runtime speeds directly against fused kernel setups.")
    else:
        print("[INFO] Fused CUDA kernels active (fast path).")
    print("=" * 60)

if __name__ == "__main__":
    print_backend_summary()
