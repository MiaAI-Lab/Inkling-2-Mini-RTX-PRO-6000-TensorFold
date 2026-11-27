<h1 align="center">Inkling 2 Mini on RTX PRO 6000 (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<p align="center">
  <img src=".github/rtx.webp" alt="Inkling 2 Mini on RTX PRO 6000 (TensorFold)" width="100%" />
</p>

Serve **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** on a single **NVIDIA RTX PRO 6000 Blackwell** (96 GiB VRAM, `sm_120`, GPU 0 by default) through an OpenAI- and Anthropic-compatible API. It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine (`tensorfold-native`, Jürgen Schmied's Inkling 2 Mini CUDA engine at [PR #4441](https://github.com/ashhart/TensorFold/pull/4441), commit `dc2ca8a1`) plus `patches/0001`–`0029` in a slim `nvidia/cuda:13.3.1-runtime-ubuntu24.04` container, with native CUDA multimodal vision (`inkling2_vision`: PNG, JPEG, WebP) and structured output / tool calling (`libtfgrammar.so`).

- **Model:** 26.4B total / 3.4B active (~5.4B effective) looped MoE (24 physical layers, middle 12 looped 3× = 48 visits; 8 of 128 experts; 40 sliding 1,024-token + 8 full-attention visits) + 23-layer `inkling2_vision` ViT encoder (`0.59 GiB` bf16). NVIDIA ModelOpt checkpoint: NVFP4 experts (layers 0–19), MXFP8 (20–23), bf16 MTP DSpark drafter (`mtp.safetensors`, `DRAFTER=dspark`).
- **Default drafter:** [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), a DFlash 2 block-diffusion draft model (5 layers, 0.49B parameters, Apache-2.0), pinned at revision `4f9088ea`, fetched into `$HOME/models/hf/Inkling-2-Mini-DFlash2` by `scripts/fetch.sh drafter` (~1 GB) and SHA-256 verified. One pass drafts a block of 7 tokens from the target's visits 5, 23, 41 and the final norm; patch `0018` runs it natively in the engine (`--drafter /models/draft`), and the target verifies every draft, so replies are unchanged.
- **Default checkpoint:** upstream `thinkingmachines/Inkling-2-Mini` (pinned rev `7a67ae23`, auto-downloaded into `$HOME/models/hf/Inkling-2-Mini` by `scripts/fetch.sh` and SHA-256 verified). Set `RERISKED=1` to serve a local [`Inkling-2-Mini-Derisked`](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked) folder instead (never downloaded automatically; behaviorally modified derivative with most safety refusals removed—treat its output as untrusted data). Blackfrost's [deployment kit](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked/blob/main/deployment-kit/AGENT_INSTRUCTIONS.md) only ships for vLLM; MiaAI-Lab built full TensorFold parity for it (matching vLLM's `reasoning_effort`, `reasoning` output field, SHA-256 checks, and readiness gate).
- **Default serving config:** GPU `0`, **262,144** context per stream (model limit 1,048,576), **8** concurrent streams (`~2.1M` active KV tokens + `16 GiB` prompt cache), `fp8` KV cache, DFlash2 drafter (`DRAFTS=1`, `DRAFTER=dflash2`), native CUDA vision (`VISION=1`), structured output / `tool_choice` (`libtfgrammar.so`), Blackwell native FP4/MXFP8 prefill tensor cores (`TF_INKLING2_PREFILL_TC=1`), and FP8 tensor-core prompt attention projections (`TF_INKLING2_ATTN_FP8_TC=1`).
- **One command:** `./start.sh` downloads weights, builds the `sm_120` image on first run, and starts the server; `./stop.sh` stops it.

## Performance

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash/) on a single **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (`sm_120`, `Inkling-2-Mini`, `CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `TF_INKLING2_ATTN_FP8=1`, `TF_INKLING2_PREFILL_TC=1`, `TF_INKLING2_ATTN_FP8_TC=1`). Prefill (patches `0001`–`0029`, DFlash2) is shown at both **300 W** and **600 W**; decode numbers below are from the earlier **300 W** (Max-Q) DSpark run.

### Prefill (300 W vs 600 W · patches `0001`–`0029`)

| Context | 300 W tok/s | 300 W TTFT | 600 W tok/s | 600 W TTFT |
| --- | ---: | ---: | ---: | ---: |
| 4k | **14,987.4 tok/s** | 325 ms | **25,137.3 tok/s** | 214 ms |
| 8k | **13,751.4 tok/s** | 647 ms | **22,466.3 tok/s** | 416 ms |
| 16k | **13,082.8 tok/s** | 1.30 s | **21,399.3 tok/s** | 817 ms |
| 32k | **12,356.0 tok/s** | 2.70 s | **20,353.2 tok/s** | 1.66 s |
| 64k | **11,035.5 tok/s** | 5.99 s | **18,354.6 tok/s** | 3.62 s |
| 128k | **9,133.2 tok/s** | 14.40 s | **15,197.5 tok/s** | 8.68 s |

> **DFlash2:** the tables below were measured with the DSpark drafter. DFlash2 (now the default) hasn't been benchmarked on the RTX PRO 6000 yet. On a DGX Spark, with the same engine and settings and only the drafter changed, it decoded 13–37% faster at one stream and 20–25% faster at 4 and 8 streams (`temperature 0.7`, thinking on). See the [DGX Spark recipe](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold#performance).

### Decode — 300 W (`temp 0.7` · thinking on · `top_k 20` · DSpark drafts)

| Streams | Prose agg | Prose / stream | Prose TTFT | Code agg | Code / stream | Code TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ×1 | **150.7 tok/s** | 150.7 tok/s | 23 ms | **231.1 tok/s** | 231.1 tok/s | 53 ms |
| ×2 | **209.6 tok/s** | 110.1 tok/s | 44 ms | **367.6 tok/s** | 194.4 tok/s | 72 ms |
| ×4 | **313.3 tok/s** | 83.5 tok/s | 62 ms | **429.8 tok/s** | 116.5 tok/s | 125 ms |
| ×8 | **471.6 tok/s** | 63.7 tok/s | 113 ms | **652.5 tok/s** | 91.9 tok/s | 234 ms |

## Checks

All checks passed on GPU 0 (`Inkling-2-Mini`, `CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `DRAFTS=1` with DSpark, patches `0001`–`0017`). Patch `0018` (DFlash2) compiles for `sm_120` on this patch stack; its checks (`exact.py` at pads 0, 3k and 24k, prompt reuse, the 200k needle, structured output 10/10, vision) passed on a DGX Spark and haven't been re-run on the RTX PRO 6000 yet:

- `tools/exact.py`: **PASS** (all 6 prompts match across serial drafted, `"draft": false`, and 6-way concurrent; 508 drafts accepted)
- `tools/prompt_reuse.py`: **PASS** (99.4%–99.5% prompt cache reuse across 4 follow-up turns at 32k; 0.06–0.07 s warm prefill)
- 200k-token Needle in a Haystack (`199,951` prompt tokens, `max_tokens=512`): **PASS** (`warm-library-4557` retrieved; 32.83 s cold prefill)
- Structured output & `tool_choice` (`json_schema`, `json_object`, `guided_regex`, `guided_choice`, `required`, named function): **PASS** (0 HTTP 500s, valid JSON / tool calls across greedy and sampled draws)
- Multimodal vision (`VISION=1`): **PASS** (PNG, RGBA checkerboard alpha, multi-image, WebP, JPEG, vision + `json_schema`, and multi-turn image prefix cache reuse)

## Requirements

- An **NVIDIA RTX PRO 6000 Blackwell** (96 GiB VRAM, `sm_120`, x86_64 Linux) with current NVIDIA drivers (CUDA 13+). Other Blackwell GPUs (`sm_120`/`sm_121`) work by adjusting `SM`, `PARALLEL`, and `CONTEXT`.
- Docker with the NVIDIA Container Toolkit, and your user in the `docker` group.
- ~15 GB free under Docker's root for the slim CUDA image, plus ~20 GB for the model checkpoint.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Inkling-2-Mini-RTX-PRO-6000-TensorFold.git
cd Inkling-2-Mini-RTX-PRO-6000-TensorFold
./start.sh
```

On the first run, `./start.sh` downloads `thinkingmachines/Inkling-2-Mini` (~18.8 GB) and `incoai/Inkling-2-Mini-DFlash2` (~1 GB), both resumable, builds `tensorfold-inkling2:c4-dc2ca8a-sm120` with embedded `sm_120` kernels, verifies the checkpoint's and the draft model's SHA-256 once, launches the server on **GPU 0** (port `8888`), and runs a smoke test. Override settings via environment variables or `.env` next to `start.sh`.

```bash
curl -s http://<host-address>:8888/v1/models

curl -s http://<host-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Inkling-2-Mini",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'
```

```bash
./start.sh restart                  # restart (validates args before stopping)
RERISKED=1 ./start.sh restart       # serve local Inkling-2-Mini-Derisked folder
GPU_ID=1 ./start.sh restart         # pin to another GPU
./stop.sh                           # stop container and free VRAM
docker logs -f inkling2-mini-tf     # follow server logs
curl -s http://<host-address>:8888/health
```

`FOREGROUND=1 ./start.sh` stays attached; `DRY_RUN=1 ./start.sh` prints the `docker run` command without changing anything; extra arguments pass straight to `tensorfold-native serve` (`./start.sh restart --context 131072`).

## KV cache & VRAM budget

On a 96 GiB RTX PRO 6000 with defaults (`CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `DRAFTS=1`, `TENSORFOLD_MEMORY_RESERVE_GIB=4`):

- **Weights & scratch:** ~19.9 GiB (`17.60 GiB` target + `1.67 GiB` DFlash2 + `0.59 GiB` vision + scratch)
- **Active KV slots:** **36.7 GiB** (`4.59 GiB`/stream × 8 streams = **2,097,152 active tokens**, `fp8` target KV + DFlash2's fixed 40 MiB ring a stream)
- **Prompt cache pool:** what's left after the `4.0 GiB` VRAM reserve, about 8 GiB more than with DSpark (DGX Spark measurement; not yet measured here)
- **`DRAFTER=dspark`:** `1.54 GiB` of drafter weights, `5.56 GiB` a stream (DSpark's global layer keeps a bf16 K/V row a position), a 16.0 GiB prompt cache, ~60.5 GiB of KV memory in all

Every stream reserves its full `CONTEXT` window at startup. Set `KV_DTYPE=bf16` (with e.g. `PARALLEL=4`) for exact bf16 KV caches; `TF_INKLING2_ATTN_FP8=0` for exact bf16 attention projections; `TF_INKLING2_PREFILL_TC=0` for exact bf16 prefill experts; `TF_INKLING2_ATTN_FP8_TC=0` for exact bf16-activation prompt attention projections.

## Configuration

All defaults live in [`scripts/config.sh`](scripts/config.sh). Override via environment (`PARALLEL=4 ./start.sh restart`) or `.env` (environment wins):

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_ID` | `0` | single GPU index passed to `--gpus device=$GPU_ID` |
| `SM` | `120` | CUDA SASS target (`120` for RTX PRO 6000 Blackwell; `121` for GB10) |
| `RERISKED` | `0` | `0`: upstream `Inkling-2-Mini` (auto-fetched); `1`: local `Inkling-2-Mini-Derisked` |
| `MODEL_DIR` | `$HOME/models/hf/Inkling-2-Mini` | checkpoint folder mounted read-only at `/models/ink` |
| `PARALLEL` | `8` | concurrent request slots (each gets a full `CONTEXT` window) |
| `CONTEXT` | `262144` | prompt + reply window per stream (up to `1048576`) |
| `KV_DTYPE` | `fp8` | target KV cache format: `fp8` (default) or `bf16` (exact) |
| `DRAFTS` | `1` | `1`: draft with `DRAFTER`; `0`: `--no-drafts` |
| `DRAFTER` | `dflash2` | `dflash2`: the DFlash2 draft model (`--drafter /models/draft`); `dspark`: the checkpoint's DSpark head (`--drafter /models/ink`) |
| `DRAFTER_DIR` | `$HOME/models/hf/Inkling-2-Mini-DFlash2` | DFlash2 folder (auto-fetched), mounted read-only at `/models/draft` |
| `VISION` | `1` | `1`: Inkling 2 vision tower (`--vision`, ~0.59 GiB weights, PNG/JPEG/WebP); `0`: text-only |
| `VISION_URLS` | `0` | `1`: also allow public HTTPS image URLs (`--vision-urls`, requires `VISION=1`) |
| `THINKING` / `REASONING_EFFORT` | `1` / unset | thinking toggle and optional default effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `0`–`0.99`) |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `1.0` / `0` | default sampling (per-request values override) |
| `MAX_TOKENS` | `32768` | default reply budget when a request omits `max_tokens` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | API listen address |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `4` | VRAM kept free when sizing the KV / prompt-cache pool |
| `TF_INKLING2_ATTN_FP8` | `1` | FP8 attention projections (`0` for exact bf16) |
| `TF_INKLING2_PREFILL_TC` | `1` | Blackwell native FP4/MXFP8 prefill tensor cores (`0` for exact bf16 MMA) |
| `TF_INKLING2_ATTN_FP8_TC` | `1` | FP8 tensor-core prompt attention projections (`0` for exact bf16 activations; decode stays bf16) |

### API notes

- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic-format), `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/metrics`, `/dashboard`.
- **Vision:** Enabled by default (`VISION=1`, `--vision`). Accepts PNG, JPEG, and WebP `image_url` data URLs (up to 100 images per request via `--vision-max-images`; set `VISION_URLS=1` to allow public HTTPS image URLs). Multi-turn follow-ups reuse cached image prefixes automatically.
- **Structured output & tool calling:** Full support for `response_format` (`json_schema`, `json_object`), `guided_regex`, `guided_choice`, and `tool_choice` (`auto`, `required`, or named function). Set `TF_TOOL_CALL_GUARD=1` to hide well-formed calls to unoffered tools (with `TF_TOOL_CALL_DEBUG=1` to log rejected envelopes).
- **Dashboard:** Pass `--dashboard` (and `--dashboard-open` to allow keyless browser access to `/dashboard` and `/stats` when `TENSORFOLD_API_KEY` is set).
- **Thinking:** The model reasons before answering even at `reasoning_effort: "none"` (give replies enough `max_tokens`). Pass `"chat_template_kwargs": {"reasoning_effort": "low"}` per request to adjust.
- **Exact serial reference:** `"draft": false` on any request disables drafting; drafted replies match `"draft": false` token-for-token.
- **Admission:** When all `PARALLEL` slots are full, new requests are rejected rather than queued.

## Patches (`patches/0001`–`0029`)

Applied in order onto TensorFold `dc2ca8a1` (`PR #4441`) by `scripts/prepare.sh`:

| Patches | Summary |
| --- | --- |
| `0001`–`0004` | Drafted verify windows with exact conv rollback & taps; CUDA DSpark drafter (`mtp.safetensors`); `--kv-cache fp8`; Inkling 2 reply/reasoning/tool parser (PR #4124) |
| `0005`–`0009` | Kernel speedups: tiled loop mix, batched DSpark holds, FP8 draft head/layers, 2-warp NVFP4 expert split, certified lazy unembedding head, warp top-k, fused sliding prompt attention & multi-row projections (all exact) |
| `0010`–`0012` | vLLM-compatible `reasoning_effort` & `reasoning` field; bf16 decode attention prefetch (`cp.async`); opt-in FP8 attention projections (`TF_INKLING2_ATTN_FP8=1`) |
| `0013` | RTX PRO 6000 / Blackwell exact speedups: 4,096-row prompt chunks (`prompt_rows = 4096`) and a 64-entry CUDA Graph cache (`wide_graphs`) for drafted / multi-row verify rounds |
| `0014` | Blackwell native block-scaled tensor-core prefill experts (`sm_120a`/`sm_121a`, `TF_INKLING2_PREFILL_TC=1`): pre-quantizes prompt activations once per layer (`38 MiB` scratch) and runs `kind::mxf4nvf4` (layers 0–19) and `kind::mxf8f6f4` (20–23) directly over TensorFold's packed weights |
| `0015` | Structured output (`response_format`, `guided_*`) and `tool_choice` (`required` / named function) on CUDA: grammar-masked greedy/sampled draws, vLLM-parity Inkling framing & tool-choice prompts, and `libtfgrammar.so` built from the patched tree with xgrammar v0.2.8 compile cache disabled |
| `0016` | Native CUDA vision encoder (`inkling2_vision`, C6): `Inkling2ImageProcessor` (PNG/JPEG/WebP, 8×8 checkerboard alpha, `-1.0` padding to 40×40 tiles), 23-layer PE-Lang ViT with 2D pairwise axial RoPE, LayerScale, bilinear pos-embed interpolation, `vision_kernel<64, 8>` tensor-core attention, 2×2 spatial merge + GELU projector, and per-slot prompt embedding splicing (`--vision`) |
| `0017` | Server & API compatibility fixes: free reply xgrammar on `Stream.init` OOM (`lane_stream.zig`), opt-in `TF_TOOL_CALL_GUARD=1` & rejected tool-envelope logging (`TF_TOOL_CALL_DEBUG=1`), raise `--vision-max-images` cap to `100`, and `--dashboard-open` (keyless `/dashboard` + `/stats` when API keys are set) |
| `0018` | DFlash2 drafter (`mini_dflash.zig`, `inkling2/dflash.cu`): `--drafter` on a DFlash2 folder loads it in place of DSpark. Committed taps go through `fc` and the hidden norm into 5 layers' 2,048-position K/V rings; one pass embeds the pending token and 7 masks and runs the two-tap dynamic convs, non-causal block attention and SwiGLU (layers as FP8 copies); an NVFP4 copy of the target head gives each row's top 16, and a GPU candidate selector walks the path (keyed draws for sampled requests). A stream alone gives the round planner each draft's chance, calibrated by what landed. After TensorFold's GLM-5.3-Flash CUDA DFlash2 and [tonyd2wild](https://github.com/tonyd2wild)'s [Inkling DFlash2 patch](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark). Identical to the DGX Spark recipe's `0018` |
| `0019`–`0027` | Decode and prefill speedups: skip expired DFlash2 prompt ingest and unread last-visit prompt rows; batched DFlash2 ingest, FP8 `fc`/`kv`, graphed holds, shared ring attention and two-pass radix top-16 selector; fused activation quantization in TC MoE up epilogue; opt-in FP8 tensor-core prompt attention projections (`TF_INKLING2_ATTN_FP8_TC=1`); async raw fp8 K/V prompt attention staging with native `e4m3x2` dequant and trimmed global `grid.z`; native `e2m1x2`/`e4m3x2` decode expert unpacking; 128×128 exact FP8 prompt projections; fused decode K/V 4-tap conv + prep |
| `0028`–`0029` | Prefill fusions and wider chunks (after [plotarmordev](https://github.com/plotarmordev)'s [PR #6](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/6) and [PR #7](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/7)): fuse prompt loop `gate_prep` into `tf_ink2_loop_gate_down_rows`, direct 4/6-plane consumer stores from `tf_ck8_128_row_split`, and 8,192-row prompt chunks |

## Tools

`tools/` talks to the running server (`API_URL`, default `http://127.0.0.1:8888`, or `PORT`):

| Script | What it checks |
| --- | --- |
| `tools/exact.py` | greedy replies: drafted, `"draft": false`, and concurrent must match bit-for-bit |
| `tools/prompt_reuse.py` | multi-turn prompt cache reuse |
| `tools/needle.py` | passphrase retrieval in a ~200k-token haystack |
| `tools/client.py` | CLI chat client and shared helpers |

## Repository layout

```
start.sh      set up (first run) and start the server on GPU_ID
stop.sh       stop it and free VRAM
scripts/      config.sh, prepare.sh, fetch.sh, banner.sh
patches/      0001–0029 applied onto TensorFold at TF_REF
tools/        verification scripts
CREDITS.md    upstream credits
NOTICE        third-party notices
```

## License

Apache-2.0, see [`LICENSE`](LICENSE). Model weights are not part of this repository: `thinkingmachines/Inkling-2-Mini` is Apache-2.0 with Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), and `Inkling-2-Mini-Derisked` is subject to upstream terms and notices.

**Third-party software in the image.** The container built by `scripts/prepare.sh` is based on NVIDIA's CUDA image (`nvidia/cuda:13.3.1-runtime-ubuntu24.04`), governed by the [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license) and [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/). By running the image you accept those terms. The Apache-2.0 license covers this repository's scripts and patches.

## Credits

- **Recipe & patches (`0001`–`0003`, `0005`–`0027`):** Created and built by **[MiaAI-Lab](https://github.com/MiaAI-Lab)** (Mia's AI Lab), including CUDA DSpark and DFlash2 drafting, `--kv-cache fp8`, decode/prefill kernel optimizations, 4,096-row chunks, wide verify CUDA graphs, Blackwell native FP4/MXFP8 prefill tensor cores, structured output / `tool_choice`, native CUDA `inkling2_vision` multimodal image support, and full TensorFold serving support for `Inkling-2-Mini-Derisked` (whose upstream [deployment kit](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked/blob/main/deployment-kit/AGENT_INSTRUCTIONS.md) only targets vLLM); benchmarked with [sparkDash](https://github.com/MiaAI-Lab/sparkDash/).
- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and `tensorfold-native` by Ash Hart ([ashhart](https://github.com/ashhart)) and contributors.
- **Inkling 2 Mini on CUDA:** Jürgen Schmied ([jschmied](https://github.com/jschmied), [PR #4441](https://github.com/ashhart/TensorFold/pull/4441), [#4640](https://github.com/ashhart/TensorFold/pull/4640)), [ajensenwaud](https://github.com/ajensenwaud) (Inkling 2 kernels & reply parser patch `0004`, #4124–#4128), [BlivionIaG](https://github.com/BlivionIaG), [sethforprivacy](https://github.com/sethforprivacy), [gilby](https://github.com/gilby), and [akol1](https://github.com/akol1).
- **DFlash2 drafter (`0018`):** [tonyd2wild](https://github.com/tonyd2wild)'s independent Inkling 2 Mini DFlash2 patch ([Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), Apache-2.0): the NVFP4 draft-head copy and the lone-stream draft chances calibrated by what landed. The draft model is [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) by Inco AI, after [DFlash](https://github.com/z-lab/dflash) (z-lab, MIT).
- **Prefill loop gate fusion, direct plane stores & 8,192-row chunks (`0028`–`0029`):** [plotarmordev](https://github.com/plotarmordev) ([PR #6](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/6) and [PR #7](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/7)).
- **Blackwell prefill & chunking ideas:** [Basalt](https://github.com/jesdga95/basalt) by [jesdga95](https://github.com/jesdga95).
- **Model & derivative:** [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines; `Inkling-2-Mini-Derisked` by Blackfrost.
- **Runtime dependencies:** [xgrammar](https://github.com/mlc-ai/xgrammar) (`libtfgrammar.so`, Apache-2.0), [Zig](https://ziglang.org), and NVIDIA CUDA.

Full list in [`CREDITS.md`](CREDITS.md).
