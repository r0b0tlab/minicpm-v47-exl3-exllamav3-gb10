# Troubleshooting

Every failure of `./run.sh` prints the stage, a code, the cause and the fix, and writes a sanitised environment
report to `.state/env-report.txt` (no host names, paths, addresses or tokens: safe to paste into an issue). The codes
below come from `scripts/errors.tsv`, which is also what the driver prints, so this table cannot drift from it.

`./run.sh --plan` shows which stages would run and why; `./run.sh --status` shows the receipts and whether the
server is healthy; `./run.sh --from <stage>` / `--only <stage>` / `--force` re-run exactly what you need.

<!-- BEGIN GENERATED:error-codes -->
| Code | Stage | Cause | Fix |
|---|---|---|---|
| <a id="e_usage"></a>`E_USAGE` | any | Unknown flag, stage name or profile. | Run `./run.sh --help`; stages are: preflight venv torch build download verify-model retire-cache serve smoke. |
| <a id="e_state"></a>`E_STATE` | any | A stage receipt in .state/ is unreadable or the state directory is not writable. | Delete the broken receipt (`rm .state/<stage>.json`) or the whole `.state/` directory; the stage simply runs again. |
| <a id="e_arch"></a>`E_ARCH` | preflight | The machine is not aarch64. This runtime (build flags, pinned torch wheel) is for the GB10 / DGX Spark. | Run on a DGX Spark / GB10. On x86-64 use the upstream ExLlamaV3 wheels with the same pack. |
| <a id="e_no_nvidia_smi"></a>`E_NO_NVIDIA_SMI` | preflight | nvidia-smi is missing or reports no GPU. | Install the NVIDIA driver: `sudo apt-get install -y nvidia-driver-580-open && sudo reboot`, then check `nvidia-smi`. |
| <a id="e_driver"></a>`E_DRIVER` | preflight | The NVIDIA driver is older than 580, which is the first branch that supports CUDA 13. | `sudo apt-get update && sudo apt-get install -y --only-upgrade nvidia-driver-580-open && sudo reboot`. |
| <a id="e_gpu_cc"></a>`E_GPU_CC` | preflight | No GPU with compute capability 12.1 (SM121, GB10). | Use a GB10 / DGX Spark. Other GPUs need a different TORCH_CUDA_ARCH_LIST and are untested. |
| <a id="e_gpu_busy"></a>`E_GPU_BUSY` | preflight | Another process is already computing on the GPU; on GB10 it shares the same 128 GB pool. | Stop it (`nvidia-smi --query-compute-apps=pid,process_name --format=csv`), or re-run with `--allow-busy` at your own risk. |
| <a id="e_toolchain"></a>`E_TOOLCHAIN` | preflight | git, gcc, g++ or curl is missing. | `sudo apt-get install -y build-essential git curl`. |
| <a id="e_python"></a>`E_PYTHON` | preflight | No Python 3.11 to 3.13 found. | `sudo apt-get install -y python3.12 python3.12-venv`, or `PYTHON=/path/to/python3.12 ./run.sh`. |
| <a id="e_venv"></a>`E_VENV` | venv | `python -m venv` failed (usually the venv/ensurepip package is missing). | `sudo apt-get install -y python3.12-venv` (match your Python), then re-run. |
| <a id="e_nvcc"></a>`E_NVCC` | preflight | nvcc 13.x was not found; it is needed once to compile the ExLlamaV3 CUDA extension. | `sudo apt-get install -y cuda-toolkit-13-0 && export CUDA_HOME=/usr/local/cuda-13.0`, or reuse a cached wheel (`CACHE_DIR=...`). |
| <a id="e_disk"></a>`E_DISK` | preflight | Not enough free disk for the model (26.4 GiB), the venv (about 14 GiB) and the build. | Free space, or point MODEL_DIR / VENV / CACHE_DIR at a larger disk. |
| <a id="e_mem"></a>`E_MEM` | preflight | MemAvailable is below what the profile needs (see docs/MEMORY.md). | Stop other memory users, or choose a smaller profile: `./run.sh --profile gb10-text-32k` / `gb10-text-16k-lowmem`. |
| <a id="e_port"></a>`E_PORT` | preflight | The listen port is already in use. | `PORT=8001 ./run.sh`, or find the owner: `ss -ltnp \| grep :8000`. |
| <a id="e_pip_torch"></a>`E_PIP_TORCH` | torch | pip could not install the pinned torch wheel from the PyTorch index. | Check network/proxy access to download.pytorch.org and re-run; the stage resumes. |
| <a id="e_pip_reqs"></a>`E_PIP_REQS` | torch | pip could not install the pinned runtime requirements. | Check network access to pypi.org; read the pip error in `.state/logs/torch.log`; re-run. |
| <a id="e_clone"></a>`E_CLONE` | build | Fetching the pinned ExLlamaV3 commit from GitHub failed. | Check network/proxy access to github.com; the fetch is resumable, re-run. |
| <a id="e_commit"></a>`E_COMMIT` | build | The fetched ExLlamaV3 tree is not the commit pinned in runtime.lock.json. | Delete `$CACHE_DIR/src/exllamav3-*` and re-run; if it persists the pin is unreachable, open an issue. |
| <a id="e_build"></a>`E_BUILD` | build | Compiling the ExLlamaV3 CUDA extension failed. | Read the tail printed above (full log `.state/logs/build.log`). Typical causes: wrong nvcc (need 13.x), out of memory (`BUILD_JOBS=4 ./run.sh`), missing g++. |
| <a id="e_import"></a>`E_IMPORT` | build | The built extension does not import, or the MiniCPM-V-4.7 architecture is not registered. | `./run.sh --force --only build` rebuilds; make sure the venv torch matches runtime.lock.json. |
| <a id="e_download"></a>`E_DOWNLOAD` | download | `hf download` failed (network, disk, or Hugging Face rate limit). | Re-run: downloads resume. On HTTP 429 wait a few minutes; lower `DL_WORKERS=2`; anonymous access is enough, no token is needed. |
| <a id="e_no_manifest"></a>`E_NO_MANIFEST` | verify-model | The model directory is not the pinned revision and no UPLOAD_MANIFEST.json is present to verify against. | Use the pinned pack (default), or put the repo's UPLOAD_MANIFEST.json next to the shards, or `SKIP_VERIFY=1` to accept the risk. |
| <a id="e_verify"></a>`E_VERIFY` | verify-model | A shard or metadata file differs in size or sha256 from the pinned manifest. | `./run.sh --force --only download` then `--only verify-model --verify-model`. If it persists, the files on disk are damaged. |
| <a id="e_retire"></a>`E_RETIRE` | retire-cache | Shard page cache could not be retired, or physical MemFree is below the model + workspace budget. | Close other large processes and re-run `./run.sh --from retire-cache`; see docs/MEMORY.md (page cache on unified memory). |
| <a id="e_load_vram"></a>`E_LOAD_VRAM` | serve | `Insufficient VRAM in split for model and cache`: the loader saw too little free memory, normally because shard page cache counted as used. | `./run.sh --from retire-cache` (retires only the shard pages), make sure nothing else uses the GPU, or pick a smaller profile. |
| <a id="e_serve_exit"></a>`E_SERVE_EXIT` | serve | The server process exited before it was ready. | Read the log tail printed above (`.state/logs/serve.log`) and `scripts/env_report.sh` output. |
| <a id="e_serve_timeout"></a>`E_SERVE_TIMEOUT` | serve | The server did not become ready within READY_TIMEOUT seconds. | Look at `.state/logs/serve.log`; a cold load of the 26 GiB pack can take minutes on slow disks: `READY_TIMEOUT=3600 ./run.sh`. |
| <a id="e_smoke"></a>`E_SMOKE` | smoke | A smoke request failed (HTTP error, wrong answer, or malformed response). | Read the failing check above; the server stays up for inspection: `curl -s http://127.0.0.1:8000/health`. |
| <a id="e_guard"></a>`E_GUARD` | serve | The optional memory guard (GUARD=1) could not start. | Unset GUARD or read `.state/logs/guard.log`. |
<!-- END GENERATED:error-codes -->

## The ones you are most likely to meet

### `Insufficient VRAM in split for model and cache` (E_LOAD_VRAM)

On GB10 the GPU and the CPU share one LPDDR5X pool and the CUDA free-memory reading that the ExLlamaV3 loader uses
counts clean file cache as used. Reading 26 GiB of shards leaves 26 GiB of page cache; on a machine whose cache is
already large the loader then concludes there is no room for weights plus KV cache, even though `MemAvailable`
looks healthy. The `retire-cache` stage calls `posix_fadvise(POSIX_FADV_DONTNEED)` on the pack's own shard files
only (never a host-wide cache drop), reads the residency back with `mincore(2)` and refuses to continue if pages
remain or physical `MemFree` is below the model plus workspace budget. If you start the server by hand, run
`python scripts/retire_page_cache.py --dir <model dir>` first. Details: [MEMORY.md](MEMORY.md).

### `nvcc` is missing or not 13.x (E_NVCC)

The CUDA extension is compiled once (about two minutes on a GB10) with `TORCH_CUDA_ARCH_LIST=12.1`. Install the
CUDA 13 toolkit (`cuda-toolkit-13-0`) and make `CUDA_HOME` point at it; the driver (>= 580) alone is not enough. The
compiled wheel is cached under `$CACHE_DIR/wheels/<commit>-<torch>-<python>-<arch>/`, so a second checkout does not
need `nvcc` at all.

### The build runs out of memory or is slow (E_BUILD)

The compile uses every core by default. `BUILD_JOBS=4 ./run.sh` limits parallelism. A cold build on the
reference GB10 took 126 s with 20 jobs.

### Port in use (E_PORT)

`PORT=8001 ./run.sh`, or find the owner with `ss -ltnp | grep :8000`. The server binds `127.0.0.1` by default.

### Hugging Face rate limits or flaky downloads (E_DOWNLOAD)

Downloads are anonymous, parallel (`DL_WORKERS`, default 8), resumable, and retried three times with a back-off.
A 429 from the Hub is retried; if it persists wait a few minutes or lower `DL_WORKERS`. No token is needed or
used (`HF_TOKEN` is unset for the download).

### A shard does not verify (E_VERIFY)

`./run.sh --force --only download` then `./run.sh --only verify-model --verify-model`. Verification compares size
and sha256 with the manifest pinned in `config/model.yaml` (per file, at the pinned revision).

### Image or video input returns 400

By design: ExLlamaV3 runs the text path of this model; the vision tower is stored (bf16) but not run. The error text
says so. Use an engine that implements the vision tower for image input.

### Thinking output ends without `</think>`

Thinking is on by default. If the model ends its turn without closing the block, the non-streaming response
returns the whole output as `content` (as the model's chat template does when it re-renders that turn).
Streaming cannot take back deltas already sent, so such a turn arrives as `reasoning_content` deltas.
`{"chat_template_kwargs": {"enable_thinking": false}}` turns thinking off.
