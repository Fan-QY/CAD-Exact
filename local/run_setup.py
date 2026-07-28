"""
Shared setup helpers.

Only the entry points used by inference and evaluation are kept in this release.
"""

import os


def attn_impl() -> str:
    """Which attention kernel to load the model with.

    flash-attn ships as prebuilt CUDA binaries with no PTX, so it cannot JIT onto
    an architecture it was not compiled for. Blackwell cards are sm_120 and this
    image's flash-attn predates them; PyTorch's own kernels can fall back through
    PTX, but flash-attn cannot. Setting CAD_ATTN=sdpa keeps the run alive on such
    a card at some cost in speed.

    Read from the environment rather than a flag because every entry point that
    loads the model must agree on it.
    """
    return os.environ.get("CAD_ATTN", "flash_attention_2")
