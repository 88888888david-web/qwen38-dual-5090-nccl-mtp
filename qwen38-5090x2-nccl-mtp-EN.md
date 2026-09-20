# Dual 5090 Running Qwen3.8-27B: Getting NCCL & MTP Right — Decode from 48 to 90 t/s (Part 2 of 1350)

**TL;DR:** This is a follow-up to my [Aug 26 post](https://lcz.me/topic/1350) titled *"Dual 5090 Running Qwen3.8-27B BF16 140K — Real Data & Optimization Notes"* (tid 1350). That post concluded: *"MTP and NCCL — two directions that 'should theoretically be faster' — actually measured slower. What won was the most vanilla tensor-split + q8_0 KV + right-sized ctx, decode ~48 t/s."*

I redid NCCL and MTP over the past month, and **the conclusion flipped**: once I made "NCCL actually running" work correctly, MTP's speculative-decoding gain finally overcame the cross-GPU sync cost. **Decode went from ~48 to ~90 t/s, ~1.9×.** The key was NOT simply "I compiled NCCL" — the 1350 post already proved "compiled NCCL but no faster." The real missing piece was **NCCL + GPU-to-GPU P2P both in place simultaneously**. Below is the full causal chain and every pitfall I hit.

## 1. Hardware & Software

| Item | Config |
| --- | --- |
| GPU | 2× RTX 5090 32GB (SM120 Blackwell) |
| CPU | Ryzen 9 9950X3D (16C/32T) |
| RAM | 60GB DDR5 |
| OS | Ubuntu 24.04.4 LTS (Kernel 7.0.0-31-generic) |
| Driver | 615.71.09 (P2P enabled, see §4.1) |
| Engine | llama.cpp **custom fork** (build 10702, commit eaf937655, `GGML_CUDA_NCCL=ON`, linked against shim NCCL 2.29.7) |
| Model | Qwen3.8-27B-BF16 (2× GGUF shards, 54.6GB total) + mmproj-F16 (928MB, Vision enabled) |
| Context | 95,000 (100K OOM in testing, 95K safe — see §4.3) |
| KV | q4_0 / q4_0 (1350 used q8_0; downgraded to q4_0 to free VRAM for MTP — see §4.4) |
| Split | tensor 0.5,0.5 |
| MTP | **Enabled** (`--spec-draft-n-max 5 --spec-draft-n-min 5`, acceptance ~48.5% — see §3) |
| Use case | Hermes Agent main brain, long-running single-slot real conversation traffic (not fixed-prompt benchmarking) |

## 2. Production Launch Parameters

Pulled directly from the running process (PID 6343) via `/proc/<pid>/cmdline` — not a sample:

```bash
# Critical: shim NCCL must load first (/usr/lib libnccl hangs on Blackwell, see §4.1)
export LD_LIBRARY_PATH="/path/to/llama-nccl/nccl-shim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

llama-server \
  -m Qwen3.8-27B-BF16-00001-of-00002.gguf \
  --mmproj mmproj-F16.gguf \
  --n-gpu-layers 99 \
  --split-mode tensor --tensor-split 0.5,0.5 \
  --ctx-size 95000 -fa on \
  --batch-size 4096 --ubatch-size 4096 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -np 1 --kv-unified \
  --jinja \
  --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-n-min 5 \
  --metrics --no-webui
```

Key notes:

- `--spec-type draft-mtp`: Qwen3.8-27B's MTP draft layer (blk.64) is **embedded in the main GGUF** — no `--model-draft` needed. This differs from Qwen3.8-Flash-Next, which requires a separate `mtp-*.gguf` head.
- `--spec-draft-n-max 5`: Guess up to 5 tokens per round. This model's draft head has low acceptance (~48.5%); going deeper gets dragged down by the low rate. 5 is the empirically balanced point.
- `-np 1 --kv-unified`: Single-slot long context. MTP only profits at single concurrency; multi-concurrency turns it net-negative (consistent with 1350's conclusion).
- `LD_LIBRARY_PATH` pointing at shim NCCL is **mandatory**: verify with `ldd` that the binary links `nccl-shim/lib/libnccl.so.2`, NOT the system version in `/usr/lib` (reason: §4.1).

## 3. Measured Data (from production logs)

All numbers come from llama-server's own `print_timing` and `/metrics`, **not** client-side measurement (client-side burst measurement runs ~30% high — a pitfall noted in 1350). Statistical window: real Agent traffic since this launch.

```text
# /metrics (cumulative)
llamacpp:tokens_predicted_total       28499
llamacpp:tokens_predicted_seconds      314.384  → decode avg 90.65 tok/s
llamacpp:prompt_tokens_total         330713   (non-cached)
llamacpp:prompt_seconds_total          113.879  → prefill avg 2904 tok/s
llamacpp:spec_decode_num_accepted_tokens_total  20187
llamacpp:spec_decode_num_draft_tokens_total     41603  → MTP acceptance 48.5%
llamacpp:n_tokens_max               95231    → max context for single task
llamacpp:requests_deferred               0
```

Per-task sampling (one session turn per task; decode from that task's `print_timing tg`, prefill from `prompt eval time` terminal line):

```text
task    prompt        prefill     decode (tg)
5998    ~2.0K         2496 t/s    85~94 t/s
7586    ~8.7K         2749 t/s    91~103 t/s
8478    ~5.1K         2476 t/s    95~107 t/s
8848    ~1.4K         1675 t/s    85~100 t/s
5902    ~94.7K *      2967 t/s    (long-context boundary)
7307    ~36.7K        3410 t/s    ~96 t/s
3675    ~34.1K        2681 t/s    (large one-shot prefill)
```

MTP acceptance (from `print_timing`'s `draft acceptance`, 23 samples):

```text
draft acceptance = 0.53438 (  342 /  640 generated), mean len =  3.67
draft acceptance = 0.76602 (  789 / 1030 generated), mean len =  4.83
draft acceptance = 0.49560 (  451 /  910 generated), mean len =  3.48
draft acceptance = 0.35262 (  573 / 1625 generated), mean len =  2.76
...
23-sample mean: 0.5251, range 0.3526 ~ 0.7660
```

Observations:

- **Overall decode ~90 t/s** (`/metrics` 90.65, per-task `tg` mean 90.40, range 65~113) — nearly double 1350's ~48 t/s.
- **Long context barely degrades**: the 94.7K task (5902) still prefills at 2967 t/s; n_tokens_max 95231; zero OOM, zero deferred throughout.
- **Prefill also went up**: 2600~3400 t/s (1350 was ~1158 t/s). NCCL's batch communication benefits prefill just as much.
- **MTP acceptance 48.5%** is on the low side (this model's draft head is weak — 1350's Qwen3.8-Flash-Next post got 69%), but mean len is stable at 3.3~3.9, and combined with low-sync-cost NCCL, speculative decoding is net positive.
- VRAM near max: GPU0 32121 / GPU1 30985 MiB (each 32607 MiB). Headroom only 0.5~1.6GB/GPU — q4_0 KV + MTP draft layer + 95K ctx is tight.

## 4. Optimization Directions Tested: What Actually Flipped This Time

### 4.1 NCCL — 1350 Said "Compiled but No Faster"; The Real Gap Was P2P

This is the core of this post. **1350's conclusion was: "Dual 5090 over PCIe using internal AllReduce; the custom-compiled NCCL version measured no faster; keep as-is."** Redoing it this time, I found the original assessment missed a piece:

- Compiling NCCL is only step one. On Blackwell SM120, the system `libnccl` in `/usr/lib` (2.18.3+cud12) **hangs during init** — it simply cannot start. So you must use the shim NCCL 2.29.7, force the binary to link the shim via `LD_LIBRARY_PATH`, and verify with `ldd` it is NOT the `/usr/lib` version.
- **What actually makes NCCL fast is GPU-to-GPU P2P being enabled.** The dual 5090s have no NVLink. The tensor-split cross-GPU all-reduce was originally going through host memory (PCIe detour). After upgrading to driver 615.71.09 + enabling P2P, `nvidia-smi topo -p2p r` returned **OK / OK** (previously it was not), so cross-GPU communication goes over direct GPU P2P, no more host detour.
- The causal chain: **P2P enabled → cross-GPU all-reduce cost drops dramatically → MTP's sync-amplification trap is avoided → MTP speculative decoding becomes net positive → decode doubles.** The root cause MTP was disabled in 1350 was exactly: "internal AllReduce over PCIe too slow + MTP's per-round serial draft forward × cross-GPU sync each round" — a multiplicative cost explosion. With P2P added, that multiplication is broken.

In one sentence: **"Compiling NCCL" and "NCCL running fast" are two different things, with a P2P gap between them.**

### 4.2 MTP — From "Abandoned in Practice" to "Re-enabled and Profitable"

Of the three reasons MTP was disabled in 1350 (acceptance ~40%, VRAM consumption, cross-GPU sync overhead eating the gains), **the first two are unchanged; what changed is the third**:

- Acceptance is still low (48.5% vs ~40% in 1350 — slightly recovered but fundamentally still a weak draft head);
- The MTP draft layer does consume VRAM (one reason KV dropped from q8_0 to q4_0 — see §4.4);
- But **once the cross-GPU sync cost is eliminated by P2P (§4.1), speculative decoding goes from net negative to net positive**: decode 48 → 90 t/s.

How to verify MTP is actually running (for anyone with the same setup): the startup log shows `creating MTP draft context against the target model`; `/metrics` `spec_decode_num_draft_tokens_total` keeps growing; `print_timing` has lines like `draft acceptance = 0.48...`. All three present = it's really running.

### 4.3 Context: 140K → 95K (Room for MTP)

1350 used 140K ctx + q8_0 KV, with ~1~2.3GB/GPU headroom. Enabling MTP means the draft layer eats VRAM, and 140K OOMs. Tested: **100K OOMs, 95K is safe**, so cut to 95K. For use cases like Hermes Agent where context rarely exceeds 100K, 95K is sufficient. For longer contexts, you'd need to lower KV quantization further or drop MTP.

### 4.4 KV: q8_0 → q4_0

1350 used q8_0. Downgraded to q4_0 this time, purely for VRAM: the MTP draft layer + 95K ctx fills the space (GPU0 down to 0.5GB headroom). q4_0's impact on decode quality is minimal (KV quantization mainly affects long-context retrieval accuracy). Trading 1.5× VRAM space for the ability to enable MTP is a good trade.

## 5. Side-by-Side with 1350 (Same Machine, Same Model)

```text
            1350 (8/26)         This post (9/20)
engine      LM Studio 2.29.0    Custom fork build 10702 + shim NCCL 2.29.7
AllReduce   internal (PCIe)     True NCCL + GPU P2P (topo -p2p r = OK/OK)
MTP         off                 on (n-max 5, acceptance 48.5%)
KV          q8_0                q4_0
ctx         140K                95K (100K OOM)
decode      ~48 t/s             ~90 t/s  (~1.9×)
prefill     ~1158 t/s           ~2904 t/s
```

The biggest reversal: **the two things abandoned in 1350 (NCCL, MTP) both became effective optimizations again this time, because P2P filled the missing puzzle piece.**

## 6. Recommended Parameters for Dual-GPU Users (Single-Slot Long Context + MTP)

```text
--split-mode tensor --tensor-split 0.5,0.5   # identical GPU models
-fa on                                        # always enable
-ctk q4_0 -ctv q4_0                           # downgrade KV to q4_0 when MTP is on
-np 1                                         # single slot (MTP only profits here)
--ctx-size <right-sized>                      # 100K OOMs in practice, 95K safe
-b 4096 -ub 4096
--jinja --metrics
--spec-type draft-mtp --spec-draft-n-max 5    # MTP (acceptance low, don't go too deep)
# Prerequisite: NCCL + P2P BOTH in place (topo -p2p r = OK/OK) before enabling MTP;
# otherwise follow 1350 and keep MTP off.
```

## 7. Conclusion

1350's "measure before optimizing" conclusion was correct, but the NCCL test sample at the time missed the P2P piece — **the real reason "compiled NCCL but no faster" was not that NCCL was useless, but that GPU-to-GPU P2P was not enabled, so cross-GPU communication was still detouring through host memory.** Once the driver upgrade + P2P were in place, NCCL finally delivered, MTP's speculative decoding turned net positive, and decode went from 48 to 90 t/s.

The biggest takeaway remains: **measure before optimizing** — but when measuring, isolate the variables cleanly. In 1350 I attributed "NCCL not faster" to NCCL itself; only this time did I realize the real variable was P2P. All the pitfalls (shim NCCL, P2P, ctx OOM, low MTP acceptance) are documented above. Hope this is useful to anyone else trying to enable MTP on dual GPUs.

All data is reproducible: engine-side logs (`print_timing`) + `/metrics`, launch parameters as shown above. Feel free to ask questions.

---
*Note: All data is from engine-side logs; client-side measurements (TTFT/burst) use a different timing window than engine-side `print_timing` and will run higher — specify the measurement method when citing. This post is a follow-up to tid 1350; hardware/model/use-case unchanged, differences concentrated in §4.*

---
## Technical Sources

| Technology / Model | Official Source |
|---|---|
| Qwen3.8-27B (model) | [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B) |
| llama.cpp (inference framework) | [GitHub](https://github.com/ggml-org/llama.cpp) |
| NCCL (multi-GPU communication) | [GitHub](https://github.com/NVIDIA/nccl) |
