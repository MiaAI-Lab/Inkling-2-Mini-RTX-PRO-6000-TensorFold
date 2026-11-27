# Credits

This repository is a thin layer of scripts and engine patches. Its own work is licensed under the Apache License 2.0
([`LICENSE`](LICENSE)). The image `scripts/prepare.sh` builds also carries TensorFold's license files from the pinned checkout.

## Model

- **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** by Thinking Machines: the looped mixture-of-experts model, its
  ModelOpt NVFP4/MXFP8 checkpoint, its chat template and its MTP (DSpark) module. Apache-2.0, with Thinking Machines'
  [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
- **[Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** by Inco AI: the DFlash 2 block-diffusion draft model
  (two-tap dynamic convs, candidate selector) this recipe drafts with by default. Apache-2.0. Read unmodified from its pinned revision.
- **[DFlash](https://github.com/z-lab/dflash)** by Jian Chen, Yesheng Liang and Zhijian Liu (z-lab, MIT): block diffusion for speculative
  decoding, the method the draft model and its serving follow; SGLang's DFlash2 implementation is the reference for the drafter's math.
- **Blackfrost**: `Inkling-2-Mini-Derisked`, the behaviorally modified derivative this recipe serves: an edited target shard with refusals
  reduced, the MTP module unchanged. Released subject to the upstream terms and notices (its README). Not part of this repository.

## Engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the TensorFold contributors:
  the Zig engine (`tensorfold-native`), its lane rounds and drafter interface, and the Inkling 2 family code. Apache-2.0 (code before 0.6.0 also MIT).
- **Jürgen Schmied** ([jschmied](https://github.com/jschmied)): the Inkling 2 Mini CUDA engine this recipe builds on, "C4"
  ([#4441](https://github.com/ashhart/TensorFold/pull/4441)), with Mini's shapes on the attention kernels (C1, #4364) and its stored NVFP4 and
  MXFP8 experts in core (C3, #4365); and the load, attention, cross-pass, slot and snapshot fixes to the lead branch
  ([#4640](https://github.com/ashhart/TensorFold/pull/4640)).
- **ajensenwaud** ([ajensenwaud](https://github.com/ajensenwaud)): Inkling 2's reply split into reasoning, text and tool calls (#4124, patch `0004`), its CUDA attention kernels (#4125), its embedding, residual mix, conv,
  router, SwiGLU and pick kernels (#4126), and Inkling 2 Small's DSpark drafter kernels ([#4128](https://github.com/ashhart/TensorFold/pull/4128)):
  the block attention, Markov context, recurrence and vocabulary kernels this recipe's drafter adapts.
- **BlivionIaG** ([BlivionIaG](https://github.com/BlivionIaG)): Inkling 2 Mini on Metal with its bundled DSpark drafter
  ([#4337](https://github.com/ashhart/TensorFold/pull/4337)), the reference for the drafter's math, taps and folded Markov head.
- **sethforprivacy** ([sethforprivacy](https://github.com/sethforprivacy)): the DSpark fixes on Metal (FP32 kappa, the 1/sqrt(128) attention scale,
  [#4447](https://github.com/ashhart/TensorFold/pull/4447)) and Mini's serving fixes.
- **TensorFold's GLM-5.3-Flash CUDA DFlash2 drafter** (TensorFold contributors): the context rings, two-tap convs, chunked block attention,
  top-16 and selector this recipe's Inkling DFlash2 drafter (patch `0018`) ports to Mini's widths.
- **tonyd2wild** ([tonyd2wild](https://github.com/tonyd2wild)): an independent Inkling 2 Mini DFlash2 patch for TensorFold
  ([Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), Apache-2.0). Patch `0018` takes its NVFP4
  draft-head copy (`fp4Of`) and its lone-stream draft chances calibrated by the lanes' Landing table.
- **gilby** ([gilby](https://github.com/gilby)): Mini's Metal kernels, DSpark prompt cache and snapshot fixes.
- **akol1** ([akol1](https://github.com/akol1)): the plan for Inkling 2 Mini on CUDA and Metal in claimable slices
  ([#4274](https://github.com/ashhart/TensorFold/pull/4274)).
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** by the MLC-AI team (Apache-2.0): `libtfgrammar.so`, from TensorFold's v1.3.3 release archive.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and contributors.
- **NVIDIA**: the CUDA containers (`nvidia/cuda:13.3.1-devel-ubuntu24.04` / `runtime`) the image is based on, and Model Optimizer
  (the checkpoint's format).
- **plotarmordev** ([plotarmordev](https://github.com/plotarmordev)): fused prompt loop `gate_prep` into `tf_ink2_loop_gate_down_rows` and direct consumer-plane stores from fused prompt attention projections ([PR #6](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/6), integrated in patch `0028` with `tf_ck8_128_row_split`), and widening prompt chunks from 4,096 to 8,192 rows ([PR #7](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold/pull/7), integrated in patch `0029`).
- **[Basalt](https://github.com/jesdga95/basalt)** by [jesdga95](https://github.com/jesdga95): Blackwell `sm_120` Inkling 2 Mini prefill and chunking ideas.

## Recipe & Patches

- **[MiaAI-Lab](https://github.com/MiaAI-Lab)** (Mia's AI Lab): this repository's scripts, RTX PRO 6000 (`sm_120`) port, [sparkDash](https://github.com/MiaAI-Lab/sparkDash/) benchmarks, full TensorFold serving support for `Inkling-2-Mini-Derisked` (whose upstream [deployment kit](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked/blob/main/deployment-kit/AGENT_INSTRUCTIONS.md) only targets vLLM: `RERISKED=1`, vLLM-parity `reasoning_effort` & `reasoning` field, SHA-256 verification, and readiness check), and the following engine patches created and built by MiaAI-Lab (all patches except `0004`, which is `ajensenwaud`'s PR #4124):
  - `0001-mini-verify-windows-rollback-taps.patch` — multi-row verify windows with exact conv rollback and drafter feature taps
  - `0002-mini-dspark-drafter.patch` — CUDA DSpark drafter (`mtp.safetensors`) with folded Markov head
  - `0003-mini-kv-cache-fp8.patch` — `--kv-cache fp8` (e4m3 K/V with per-position/head fp32 scales)
  - `0005-mini-speed.patch` — tiled loop mix, batched DSpark holds, and decode/prefill kernel speedups
  - `0006-mini-drafts.patch` — FP8 draft head/phi and keyed sampled drafting
  - `0007-mini-expert-split.patch` — 2-warp NVFP4 decode expert split
  - `0008-mini-lazy-head.patch` — certified lazy unembedding head, warp top-k, FP8 drafter layers, and global attention z-capping
  - `0009-mini-prompt-attention-fused-rows.patch` — fused sliding prompt attention and fused multi-row input projections
  - `0010-inkling-thinking-effort-reasoning-field.patch` — vLLM-compatible `reasoning_effort` and `reasoning` output field
  - `0011-mini-attention-prefetch.patch` — bf16 decode attention `cp.async` K/V tile prefetch
  - `0012-mini-attention-fp8-optin.patch` — opt-in FP8 attention projections (`TF_INKLING2_ATTN_FP8`)
  - `0013-mini-rtx6000-sm120.patch` — 4,096-row prompt chunks and CUDA Graph cache (`wide_graphs`) for drafted / multi-row verify rounds
  - `0014-mini-prefill-tc-sm120a.patch` — Blackwell native block-scaled FP4/MXFP8 tensor-core prefill experts (`TF_INKLING2_PREFILL_TC`) with pre-quantized activations
  - `0015-mini-structured-output.patch` — structured output (`response_format`, `guided_*`) and `tool_choice` (`required` / named function) with grammar-masked draws and patched `libtfgrammar.so` (xgrammar v0.2.8 cache fix)
  - `0016-mini-vision.patch` — native CUDA `inkling2_vision` multimodal image encoder (`Inkling2ImageProcessor`, 23-layer PE-Lang ViT with 2D axial RoPE, LayerScale, `vision_kernel<64, 8>`, 2×2 spatial merge + GELU projector, and `--vision` server integration)
  - `0017-server-compat-fixes.patch` — server/API compatibility bundle: reply grammar OOM leak fix (adapted from SvangenStudios' PR #28), opt-in `TF_TOOL_CALL_GUARD=1` & rejected tool-envelope logging (`TF_TOOL_CALL_DEBUG=1`), `--vision-max-images` raised to 100, and `--dashboard-open`
