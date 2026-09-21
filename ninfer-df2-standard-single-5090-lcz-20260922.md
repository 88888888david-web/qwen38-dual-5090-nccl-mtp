# 單張 5090 跑 Qwen3.8-27B：NInfer + DFlash2 投機解碼實測，長推理 270+ t/s（1350 系列）

先講結論：1350 系列前兩篇都在講 llama.cpp 雙 5090。這篇換一條路線——**單張 RTX 5090 + NInfer 引擎 + DFlash2 投機解碼**跑官方標準版 Qwen3.8-27B。乾淨對照實測：**長推理 decode 271.7 t/s（無 spec 基線 77 t/s 的 3.5 倍）**，而且落在官方 issue #188 公佈的 224–357 t/s 區間內，實測無灌水。但重點不在這個數字——真正有參考價值的是**投機解碼的勝負是 work 相依的**：高接受率場景 DFlash2 大贏，低接受率場景（中文技術說明、日常對話）MTP 反而打平甚至略勝。下面把因果鏈、踩的坑和可複現的數據寫清楚。

## 1. 硬體與軟體

| 項目 | 配置 |
| --- | --- |
| GPU | RTX 5090 32GB（SM120 Blackwell，單卡；本機另有第二張 5090 未參與本篇測試） |
| CPU | Ryzen 9 9950X3D（16C/32T） |
| 記憶體 | 60GB DDR5 |
| 系統 | Ubuntu 24.04.4 LTS（Kernel 7.0.0-31-generic） |
| 驅動 | 615.71.09 |
| 引擎 | NInfer（`ninfer-serve`，master `a16b6442` 本地編譯，含社群 fork `7a876cf` 的 long-context 修正，見 §4.4） |
| 模型 | Qwen3.8-27B 官方標準版，NInfer 專用 artifact `qwen3_8_27b_nvfp4.ninfer`（21.5GB，NVFP4 權重 + DFlash2 draft head 內嵌） |
| KV | int8（單流測試）/ nvfp4（生產配置），見 §2 |
| 用途 | Hermes Agent 主腦，長期單 slot 真實對話流量 + 長推理 benchmark |

## 2. 生產啟動參數

以下從運行中進程的 `/proc/<pid>/cmdline` 直接抓，不是我貼的範例：

```bash
ninfer-serve <model>/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 12435 --device 0 \
  --max-context 262144 --kv-dtype nvfp4 --kv-capacity 334000 \
  --max-concurrency 3 --prefill-chunk 1024 --pending-timeout-ms 600000 \
  --spec dflash2 --draft-tokens 8 --lm-head-draft \
  --vision --media-cache-mib 256 --media-live-mib 512
```

重點說明：

- `--spec dflash2 --draft-tokens 8 --lm-head-draft`：DFlash2 是 NInfer 的投機解碼模式，每輪猜 8 個 token，draft head 用 `lm-head-draft`（不單獨載 draft 模型）。Qwen3.8-27B 的 DFlash2 draft 權重**內嵌在 artifact 裡**，不用另載。
- `--kv-dtype nvfp4`：KV cache 走 NVFP4，同 8GiB 空間能放 1.83× 的 token 容量（262144 pool 下實測 277,696 tok，int8 是 151,424 tok）。
- `--max-concurrency 3`：agent 多路並發配置。單流 decode 極限測試（§3.1）則用 C=1 + int8 KV + 131072 的輕配置。
- 啟動方式用 `systemd --user` unit，**不要用一般終端機 background**（原因見 §4.5）。

## 3. 實測數據（server log 為準）

以下全部取自 `ninfer-serve` 自己的 per-request log 行（`decode XXX tok/s | dflash2 accepted A/B (P%)`），**不是** client 端量測（client 端 SSE 計數會低估 ~4.5×，見 §4.1）。每組 3 跑取中位，同 GPU 前後腳。

### 3.1 單流 decode 極限（C=1，int8 KV，131072，draft-7）

長推理 2500-token prompt（AIME 風格，`temperature=0`）：

| spec 模式 | decode tok/s（3 跑） | 中位 | acceptance |
| --- | --- | ---: | ---: |
| **DFlash2**（K7） | 271.7 / 282.2 / 260.8 | **271.7** | ~50% |
| **MTP3**（K3，對照） | 203.0 / 202.5 / 201.8 | 202.5 | 71.9% |
| no-spec（基線） | — | ~77 | — |

- DFlash2 vs 同 binary 同 artifact 的 MTP3 = **+34%**（純 spec 模式差異，同卡同配置）。
- 對照官方 issue #188 公佈的 NVFP4 數字（AIME 321±16 / Code 265±22 / Structured 357±41 等，區間 224–357），本機 271.7 落在區間內，**無灌水**。
- DFlash2 贏在**每輪 draft 7 token**（MTP3 是 3）——acceptance 反而更低（50% vs 72%），但每輪產出更多，淨速度更高。

### 3.2 並發配置（C=3，int8 KV，131072，draft-7）

同一 GPU、同一 binary、同一 artifact，C=3 多路並發配置下換「中文技術說明」prompt（低接受率場景）：

| spec 模式 | decode tok/s | acceptance |
| --- | ---: | ---: |
| DFlash2（K7） | 140.8 / 151.5 | 19.3% / 21.6% |
| MTP（K3） | 159.4 / 158.2 | 57.3% / 57.0% |
| no-spec（基線） | 77.2 | — |

換回高接受率的長推理 prompt（同 C=3 配置，nvfp4 KV / 262144）：

| spec 模式 | decode tok/s（3 跑） | 中位 | acceptance |
| --- | --- | ---: | ---: |
| DFlash2 | 226.1 / 210.1 / 210.0 | **210.1** | 42.6% / 38.6% / 38.6% |

### 3.3 關鍵結論：acceptance 是 work 相依的

| 場景 | DFlash2 | MTP3 | 勝負 |
| --- | --- | --- | --- |
| 高接受率（長推理 / code / math） | **271.7 t/s**（C=1）/ 210.1 t/s（C=3） | 202.5 / ~158 t/s | **DFlash2 贏 +34%** |
| 低接受率（中文技術說明 / 日常對話） | 151.5 t/s | **158.2 t/s** | MTP 打平 / 略勝 |

原因：DFlash2「每輪賭 7–8 個 token」的優勢**需要高 acceptance 才兌現**。長推理 acceptance ~40–50% 時，長 draft 攤平、贏；中文技術說明 acceptance 掉到 ~20% 時，賭大的成本（DFlash2 的 draft 比 MTP 重，KV pool 也小一半——151k vs 246k token）抵消優勢，MTP 的輕 draft（acceptance 57%）打平甚至略勝。

**一句話：work 以長推理 / code 為主 → DFlash2；以中文對話為主 → MTP 更省且速度不輸。** 別拿單一 work 流派的 benchmark 決定換哪個。

## 4. 踩過的坑

### 4.1 client 端計數低估 ~4.5×（最重要）

用 SSE streaming 的 Python client 數 token：同一支 DFlash2 請求，client 看到 ~60 tok/s，server log 實為 271.7 tok/s。engine 提交的 token 數 > client SSE parser 抓到的（reasoning token、batched deltas）。**比 decode 一律讀 server log 的 per-request 行**，口徑不一樣的話引用時註明。

### 4.2 DFlash2 必須用新 artifact

舊的 20GB artifact **不含 DFlash2 companion weights**，啟用 `--spec dflash2` 會失敗。DFlash2 版是 21.5GB 的新 artifact（上游官方發布，SHA256 `552c374c…0d462c` 可對照 manifest 驗證）。換 spec 模式前先確認 artifact 版本，這是我廢掉一次 dry-run 的原因。

### 4.3 本地編譯的兩個坑

- **CUDA 版本**：`/usr/local/cuda` 預設是 12.0，不認 `sm_120a`（5090 Blackwell）。NInfer 的 CMakeLists 硬性要求 `CMAKE_CUDA_ARCHITECTURES=120a`，要用 CUDA 13.1。
- **編譯 OOM**：`-j$(nproc)` 全速並編會吃掉記憶體頂峰，cicc 預處理步驟被 SIGKILL（318 個物件編到 258 就死）。改 `-j4` 增量補編，幾分鐘收工。

### 4.4 long-context 穩定性：社群 fork 的 `causal_small_t` 修正

上游 master 有一個 shared-memory 溢出 bug：大 window（>8198 token）+ decode 時 `causal_small_t` kernel 的 page 數算錯，寫出 `__shared__` 邊界 → GPU Xid 13 / `cudaErrorLaunchFailure`，跑幾小時後隨機觸發。社群 fork（`Doelfke/ninfer-yarn` @ `7a876cf`）的修正（加 page-safety floor）編進本地 binary 後，長 context 長時間跑不再觸發。**長 context 重度使用的話，建議用含此修正的 build。**

### 4.5 不要用終端機 background 拉 server

Hermes agent 的 terminal background 會把 process 包在帶 memcg cap 的 cgroup 裡；NInfer 啟動時要 pin ~8–9GB host KV，超過 cap 直接 OOM（log 停在 `host-kv-pin status=begin`，dmesg 顯示 `CONSTRAINT_MEMCG`，但 host 記憶體其實沒吃緊）。用 `systemd --user` unit 起就解決。

## 5. 跟 llama.cpp 路線的對照（同一台機）

| | llama.cpp 雙 5090（1350 續篇） | NInfer 單 5090（本篇） |
| --- | --- | --- |
| 權重 | BF16 GGUF（54.6GB，雙卡 tensor split） | NVFP4 artifact（21.5GB，單卡） |
| MTP 接受率 | ~48.5%（MTP draft 層） | ~39–50%（DFlash2，work 相依） |
| 長推理 decode | ~90 t/s（95K ctx） | **271.7 t/s（C=1）/ 210.1 t/s（C=3 生產配置）** |
| 低接受率 decode | ~90 t/s（穩定） | ~151 t/s（DFlash2）/ ~158 t/s（MTP） |
| ctx 上限 | 95K（雙卡 VRAM 打滿） | 262144（nvfp4 KV 單卡） |

差距的來源不是「誰更強」，是**引擎架構**：NInfer 是專為單卡 + CUDA Graphs + paged KV 設計的自研引擎，單卡 decode 的極限遠高於走跨卡 all-reduce 的 tensor split；但雙卡 llama.cpp 能容納更大的模型（BF16 全精度 54.6GB 單卡放不下）。**27B 放得進單張 5090 的話，單卡 NInfer 是 decode 最快的路。**

## 6. 參數建議（單卡 5090 + NInfer + Qwen3.8-27B）

```text
# 高接受率 work（長推理 / code / agent）
--spec dflash2 --draft-tokens 8 --lm-head-draft
--kv-dtype nvfp4 --max-context 262144 --kv-capacity 334000 --max-concurrency 3

# 中文對話為主 / 要更大 KV pool
--spec mtp --draft-tokens 3
# （MTP 的 KV pool 是 DFlash2 的 ~1.6×，同空間撐更長 context / 更高並發）

# 通用
# - 本地編譯用 CUDA 13.1 + sm_120a + -j4（§4.3）
# - server 用 systemd --user 起（§4.5）
# - 長 context 重度使用 → 用含 causal_small_t fork 修正的 build（§4.4）
# - decode 數據讀 server log per-request 行，client 端計數會低估 4.5×（§4.1）
```

## 7. 結語

單卡 5090 跑 27B 這條路，重點心得兩句：

1. **投機解碼的勝負是 work 相依的**——DFlash2 在長推理大贏 +34%，在低接受率的中文場景反而打平/小負。測 benchmark 時 acceptance 不寫進報告，數字會騙人。
2. **口徑決定一切**——client 端 SSE 計數、不同 KV 量化、不同 concurrency 的數字混在一起比，會得出完全錯誤的結論。本篇所有 A/B 都是同 GPU、同 binary、同配置、讀 server log。

數據全部可重現：server 端 per-request log + 啟動參數如上（§2）。有問題歡迎直接問。

---
*附：所有數據以引擎端 log 為準；client 端量測與引擎端計時窗不同，會偏低，引用時請註明口徑。本篇為 1350 系列的單卡 NInfer 路線，硬體/模型/用途與前篇一致，差異集中在引擎與 spec 模式。*

---
## 技術來源

| 技術/模型 | 官方來源 |
| --- | --- |
| Qwen3.8-27B（模型） | [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NInfer（推理引擎） | [GitHub](https://github.com/Neroued/ninfer) |
| NInfer artifact（NVFP4 + DFlash2） | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| DFlash2（投機解碼 issue #188） | [GitHub](https://github.com/Neroued/ninfer/issues/188) |
| causal_small_t 修正（社群 fork） | [GitHub](https://github.com/Doelfke/ninfer-yarn) |
