# Running Qwen3.8-27B on a Single 5090: NInfer + DFlash2 Speculative Decoding, 270+ t/s on Long Reasoning (1350 series)

Bottom line first: the first two posts in the 1350 series were all about llama.cpp with dual 5090s. This one takes a different route — **a single RTX 5090 + the NInfer engine + DFlash2 speculative decoding** running the official standard Qwen3.8-27B. Clean head-to-head measurement: **long-reasoning decode 271.7 t/s (3.5× the 77 t/s no-spec baseline)**, and it lands inside the 224–357 t/s range the maintainers published in issue #188 — measured, no inflation. But the number isn't the point. What's actually useful is that **the outcome of speculative decoding is workload-dependent**: DFlash2 wins big in high-acceptance scenarios, while in low-acceptance scenarios (Chinese technical explanation, everyday chat) MTP ties it or even edges ahead. Below is the causal chain, the pitfalls, and the reproducible data.

## 1. Hardware & software

| Item | Setup |
| --- | --- |
| GPU | RTX 5090 32GB (SM120 Blackwell, single card; a second 5090 is present but not used in this test) |
| CPU | Ryzen 9 9950X3D (16C/32T) |
| Memory | 60GB DDR5 |
| OS | Ubuntu 24.04.4 LTS (Kernel 7.0.0-31-generic) |
| Driver | 615.71.09 |
| Engine | NInfer (`ninfer-serve`, master `a16b6442` compiled locally, with the community fork `7a876cf` long-context fix included, see §4.4) |
| Model | Qwen3.8-27B official standard, NInfer artifact `qwen3_8_27b_nvfp4.ninfer` (21.5GB, NVFP4 weights + embedded DFlash2 draft head) |
| KV | int8 (single-stream test) / nvfp4 (production config), see §2 |
| Use | Hermes Agent main brain, long-term single-slot real conversational traffic + long-reasoning benchmark |

## 2. Production launch parameters

The following was captured directly from the running process's `/proc/<pid>/cmdline` — not a pasted example:

```bash
ninfer-serve <model>/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 12435 --device 0 \
  --max-context 262144 --kv-dtype nvfp4 --kv-capacity 334000 \
  --max-concurrency 3 --prefill-chunk 1024 --pending-timeout-ms 600000 \
  --spec dflash2 --draft-tokens 8 --lm-head-draft \
  --vision --media-cache-mib 256 --media-live-mib 512
```

Key notes:

- `--spec dflash2 --draft-tokens 8 --lm-head-draft`: DFlash2 is NInfer's speculative-decoding mode; it drafts 8 tokens per round, and the draft head uses `lm-head-draft` (no separate draft model loaded). Qwen3.8-27B's DFlash2 draft weights are **embedded in the artifact** — no extra load.
- `--kv-dtype nvfp4`: the KV cache uses NVFP4, so the same 8GiB of space holds 1.83× the tokens (measured 277,696 tok under the 262144 pool vs 151,424 tok for int8).
- `--max-concurrency 3`: multi-stream agent config. The single-stream decode ceiling test (§3.1) instead uses the lighter C=1 + int8 KV + 131072 config.
- The server is started via a `systemd --user` unit — **do not use a plain terminal background** (reason in §4.5).

## 3. Measured data (server log is authoritative)

Everything below is taken from `ninfer-serve`'s own per-request log lines (`decode XXX tok/s | dflash2 accepted A/B (P%)`), **not** from client-side measurement (client SSE counting underestimates by ~4.5×, see §4.1). Three runs per group, median reported, same GPU back-to-back.

### 3.1 Single-stream decode ceiling (C=1, int8 KV, 131072, draft-7)

Long-reasoning 2500-token prompt (AIME-style, `temperature=0`):

| Spec mode | decode tok/s (3 runs) | Median | Acceptance |
| --- | --- | ---: | ---: |
| **DFlash2** (K7) | 271.7 / 282.2 / 260.8 | **271.7** | ~50% |
| **MTP3** (K3, control) | 203.0 / 202.5 / 201.8 | 202.5 | 71.9% |
| no-spec (baseline) | — | ~77 | — |

- DFlash2 vs MTP3 on the same binary and same artifact = **+34%** (pure spec-mode difference, same card, same config).
- Compared to the official NVFP4 numbers in issue #188 (AIME 321±16 / Code 265±22 / Structured 357±41, etc.; range 224–357), the local 271.7 falls inside the range — **no inflation**.
- DFlash2 wins by **drafting 7 tokens per round** (MTP3 does 3) — its acceptance is actually lower (50% vs 72%), but it produces more per round, so net speed is higher.

### 3.2 Concurrency config (C=3, int8 KV, 131072, draft-7)

Same GPU, same binary, same artifact; under the C=3 multi-stream config, switching to a "Chinese technical explanation" prompt (low-acceptance scenario):

| Spec mode | decode tok/s | Acceptance |
| --- | ---: | ---: |
| DFlash2 (K7) | 140.8 / 151.5 | 19.3% / 21.6% |
| MTP (K3) | 159.4 / 158.2 | 57.3% / 57.0% |
| no-spec (baseline) | 77.2 | — |

Switching back to the high-acceptance long-reasoning prompt (same C=3 config, nvfp4 KV / 262144):

| Spec mode | decode tok/s (3 runs) | Median | Acceptance |
| --- | --- | ---: | ---: |
| DFlash2 | 226.1 / 210.1 / 210.0 | **210.1** | 42.6% / 38.6% / 38.6% |

### 3.3 Key conclusion: acceptance is workload-dependent

| Scenario | DFlash2 | MTP3 | Outcome |
| --- | --- | --- | --- |
| High acceptance (long reasoning / code / math) | **271.7 t/s** (C=1) / 210.1 t/s (C=3) | 202.5 / ~158 t/s | **DFlash2 wins +34%** |
| Low acceptance (Chinese technical explanation / everyday chat) | 151.5 t/s | **158.2 t/s** | MTP ties / edges ahead |

Why: DFlash2's "bet 7–8 tokens per round" advantage **only pays off at high acceptance**. At long-reasoning acceptance of ~40–50%, the long draft amortizes and it wins; when Chinese technical explanation drops acceptance to ~20%, the cost of the bigger bet (DFlash2's draft is heavier than MTP's, and its KV pool is half as big — 151k vs 246k tokens) cancels the advantage, so MTP's lighter draft (57% acceptance) ties or even edges ahead.

**In one line: workload is mostly long reasoning / code → DFlash2; mostly Chinese chat → MTP is leaner and not slower.** Don't pick one based on a single workload family's benchmark.

## 4. Pitfalls hit

### 4.1 Client-side counting underestimates ~4.5× (most important)

Counting tokens with an SSE-streaming Python client: on the same DFlash2 request, the client sees ~60 tok/s while the server log says 271.7 tok/s. The tokens the engine commits > what the client SSE parser catches (reasoning tokens, batched deltas). **Always read the server log's per-request lines for decode comparison**; if the measurement basis differs, state it when citing.

### 4.2 DFlash2 requires the new artifact

The old 20GB artifact **does not contain the DFlash2 companion weights**, so enabling `--spec dflash2` fails. The DFlash2 version is a new 21.5GB artifact (published by the maintainers; SHA256 `552c374c…0d462c`, verifiable against the manifest). Confirm the artifact version before switching spec modes — this is what invalidated one of my dry runs.

### 4.3 Two local-compile pitfalls

- **CUDA version**: `/usr/local/cuda` defaults to 12.0, which does not recognize `sm_120a` (5090 Blackwell). NInfer's CMakeLists hard-requires `CMAKE_CUDA_ARCHITECTURES=120a`; use CUDA 13.1.
- **Compile OOM**: a full parallel build with `-j$(nproc)` spikes memory to the ceiling and the cicc preprocessing step gets SIGKILLed (died at 258 of 318 objects). Switching to `-j4` and building incrementally finished it in minutes.

### 4.4 Long-context stability: the community fork's `causal_small_t` fix

Upstream master has a shared-memory overflow bug: with a large window (>8198 tokens) plus decode, the `causal_small_t` kernel miscalculates the page count and writes past the `__shared__` boundary → GPU Xid 13 / `cudaErrorLaunchFailure`, triggered randomly after a few hours of running. The community fork (`Doelfke/ninfer-yarn` @ `7a876cf`) fix (a page-safety floor), compiled into the local binary, stops long-context long runs from triggering it. **If you use long context heavily, use a build that includes this fix.**

### 4.5 Do not start the server with a terminal background

Hermes agent's terminal background wraps the process in a cgroup with a memcg cap; at startup NInfer pins ~8–9GB of host KV, and exceeding the cap OOMs it outright (log stops at `host-kv-pin status=begin`, dmesg shows `CONSTRAINT_MEMCG`, even though host memory is actually fine). Starting via a `systemd --user` unit resolves it.

## 5. Comparison with the llama.cpp route (same machine)

| | llama.cpp dual 5090 (1350 follow-up) | NInfer single 5090 (this post) |
| --- | --- | --- |
| Weights | BF16 GGUF (54.6GB, dual-card tensor split) | NVFP4 artifact (21.5GB, single card) |
| MTP acceptance | ~48.5% (MTP draft layer) | ~39–50% (DFlash2, workload-dependent) |
| Long-reasoning decode | ~90 t/s (95K ctx) | **271.7 t/s (C=1) / 210.1 t/s (C=3 production config)** |
| Low-acceptance decode | ~90 t/s (stable) | ~151 t/s (DFlash2) / ~158 t/s (MTP) |
| Context ceiling | 95K (dual-card VRAM full) | 262144 (nvfp4 KV, single card) |

The gap isn't "who's stronger" — it's **engine architecture**: NInfer is a purpose-built engine designed for single card + CUDA Graphs + paged KV, so single-card decode ceiling is far higher than tensor split that crosses cards via all-reduce; but dual-card llama.cpp fits bigger models (BF16 full-precision 54.6GB won't fit on a single card). **If 27B fits on a single 5090, single-card NInfer is the fastest path for decode.**

## 6. Parameter recommendations (single 5090 + NInfer + Qwen3.8-27B)

```text
# High-acceptance work (long reasoning / code / agent)
--spec dflash2 --draft-tokens 8 --lm-head-draft
--kv-dtype nvfp4 --max-context 262144 --kv-capacity 334000 --max-concurrency 3

# Mostly Chinese chat / want a larger KV pool
--spec mtp --draft-tokens 3
# (MTP's KV pool is ~1.6× DFlash2's; same space holds longer context / higher concurrency)

# General
# - Local compile: CUDA 13.1 + sm_120a + -j4 (§4.3)
# - Start server via systemd --user (§4.5)
# - Heavy long-context use → use a build with the causal_small_t fork fix (§4.4)
# - Read decode data from server log per-request lines; client counting underestimates 4.5× (§4.1)
```

## 7. Closing

Running 27B on a single 5090, two key takeaways:

1. **The outcome of speculative decoding is workload-dependent** — DFlash2 wins big on long reasoning (+34%) but ties/loses on low-acceptance Chinese scenarios. If you don't put acceptance in the benchmark report, the numbers will lie.
2. **Basis is everything** — mixing client SSE counts, different KV quantizations, and different concurrency levels into one comparison leads to completely wrong conclusions. Every A/B here is same GPU, same binary, same config, reading the server log.

All data is reproducible: server-side per-request logs + launch parameters as shown above (§2). Feel free to ask questions directly.

---
*Note: all data is authoritative from the engine-side log; client-side measurement uses a different timing window and runs lower, so state the basis when citing. This post is the single-card NInfer route of the 1350 series; hardware/model/use match the previous posts, and the difference is concentrated in the engine and spec mode.*

---
## Technical sources

| Tech / model | Official source |
| --- | --- |
| Qwen3.8-27B (model) | [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NInfer (inference engine) | [GitHub](https://github.com/Neroued/ninfer) |
| NInfer artifact (NVFP4 + DFlash2) | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| DFlash2 (speculative decoding, issue #188) | [GitHub](https://github.com/Neroued/ninfer/issues/188) |
| causal_small_t fix (community fork) | [GitHub](https://github.com/Doelfke/ninfer-yarn) |
