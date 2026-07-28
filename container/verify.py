"""Check a container against the card it is running on.

Run this on BOTH card types before switching any real run over -- an image that
works on one architecture and not the other is exactly the failure that cost us
the first Blackwell attempt, and it only shows up at the first kernel launch.

    singularity exec --nv -B /scratch:/scratch cadllm2.sif python container/verify.py

Exit status is 0 only if everything the training path needs actually works on
this GPU. Fallback-covered pieces (pytorch3d, flash-attn) report as OPTIONAL and
do not fail the check.
"""
import importlib
import sys
import traceback

ok = True
optional_missing = []


def check(label, fn, optional=False):
    global ok
    try:
        result = fn()
        print(f"  PASS      {label}: {result}")
    except Exception as e:
        if optional:
            optional_missing.append(label)
            print(f"  OPTIONAL  {label}: {type(e).__name__}: {str(e)[:100]}")
        else:
            ok = False
            print(f"  FAIL      {label}: {type(e).__name__}: {str(e)[:200]}")


def main():
    import torch

    print("== environment ==")
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"  arch list {torch.cuda.get_arch_list()}")
    if not torch.cuda.is_available():
        print("  FAIL      no CUDA device visible -- was --nv passed, and --gres?")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"  device {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")

    # The old image listed sm_90 as its maximum and carried no PTX, so this is
    # the single most important line: does this build cover the card in front of
    # it, either natively or through PTX?
    tag = f"sm_{cap[0]}{cap[1]}"
    covered = tag in torch.cuda.get_arch_list()
    ptx = any(a.startswith("compute_") for a in torch.cuda.get_arch_list())
    print(f"  {tag} in arch list: {covered}   PTX fallback present: {ptx}")

    print("== kernels ==")

    def matmul():
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        return f"bf16 matmul -> {(x @ x).float().sum().item():.3e}"

    def sdpa():
        import torch.nn.functional as F
        q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
        return tuple(F.scaled_dot_product_attention(q, q, q).shape)

    def backward():
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (w @ w).sum().backward()
        return "autograd ok"

    check("bf16 matmul", matmul)
    check("scaled_dot_product_attention", sdpa)
    check("backward pass", backward)

    print("== packages the training path imports ==")
    for m in ["torchvision", "transformers", "peft", "accelerate", "pyarrow",
              "numpy", "trimesh", "open3d", "skimage", "scipy"]:
        check(m, lambda m=m: getattr(importlib.import_module(m), "__version__", "ok"))

    print("== packages with verified fallbacks ==")

    def p3d():
        from pytorch3d.ops import sample_farthest_points
        pts = torch.randn(1, 512, 3)
        return f"fps -> {sample_farthest_points(pts, K=16)[1].shape}"

    def fa():
        from flash_attn import flash_attn_func
        q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        return tuple(flash_attn_func(q, q, q).shape)

    check("pytorch3d (else CAD_FPS_TORCH=1)", p3d, optional=True)
    check("flash-attn (else CAD_ATTN=sdpa)", fa, optional=True)

    print("== inference/eval only ==")
    check("cadquery", lambda: getattr(importlib.import_module("cadquery"), "__version__", "ok"),
          optional=True)

    print()
    if optional_missing:
        print("set these when running:")
        if any("pytorch3d" in m for m in optional_missing):
            print("  CAD_FPS_TORCH=1   (torch FPS; verified to select identical points)")
        if any("flash-attn" in m for m in optional_missing):
            print("  CAD_ATTN=sdpa     (PyTorch attention; works on sm_120)")
    print("RESULT:", "usable on this card" if ok else "NOT usable on this card")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
