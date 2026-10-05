# Environment

- venv: /mnt/storage/garv901/regime-batching/.venv (Python 3.12.13, uv). Activate: `source /mnt/storage/garv901/regime-batching/.venv/bin/activate`
- HF_HOME=/mnt/storage/garv901/.cache/huggingface
- Python 3.12 is fine (vLLM supports 3.10-3.14).

## Versions
| pkg | version |
|---|---|
| torch | 2.13.0+cu130 (torch.version.cuda 13.0) |
| vllm | 0.30.0 |
| flashinfer-python | 0.6.18.post1 (pinned by vllm) |
| flashinfer-cubin | 0.6.18.post1 (from https://flashinfer.ai/whl/ ; installed --no-deps) |
| flashinfer-jit-cache | 0.6.18.post1+cu130 (from https://flashinfer.ai/whl/cu130/ ; --no-deps) |
| triton | 3.7.1 |
Extras: pandas pyarrow matplotlib seaborn plotly scikit-learn aiohttp nvtx datasets tqdm pyyaml.
`VLLM_HAS_FLASHINFER_CUBIN=1` may be set (cubin needed for XQA decode kernels on Hopper).
Import checks (CPU): torch, vllm, flashinfer, flashinfer_cubin, `from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata` all OK.

## Nsight
- nsys (default): /usr/local/bin/nsys -> /opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys, version 2025.5.2.266
- other nsys: /opt/nvidia/nsight-systems/2025.1.1/target-linux-x64/nsys, /usr/local/cuda-{13.1,12.4,12.1}/bin/nsys
- ncu: /home/garv901/.local/bin/ncu -> /opt/nvidia/nsight-compute/2025.3.1/ncu, version 2025.3.1.0
- other ncu: /usr/local/cuda-{13.1,12.4,12.1}/bin/ncu

## Cloned repos
- third_party/vllm: tag v0.30.0 (shallow)
- third_party/vllm-flash-attention: https://github.com/vllm-project/flash-attention @ 506341a143fcabd4bb79052a7605ada727d6b3f5 (pinned in cmake/external_projects/vllm_flash_attn.cmake)

## Problems
None. https://flashinfer.ai/whl/cu130/flashinfer-cubin/ 404s but cubin installed from https://flashinfer.ai/whl/ index.
