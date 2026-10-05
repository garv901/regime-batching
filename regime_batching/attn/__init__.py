"""Regime-aware attention dispatch (component C3)."""


def register_backend():
    """Override FLASH_ATTN with the regime-aware subclass, so the default backend choice picks it up."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend as _reg

    _reg(AttentionBackendEnum.FLASH_ATTN, "regime_batching.attn.backend.RegimeFlashAttentionBackend")
