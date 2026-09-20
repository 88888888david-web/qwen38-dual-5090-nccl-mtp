# 雙 5090 跑 Qwen3.8-27B：把 NCCL 和 MTP 做對，decode 從 48 翻倍到 90 t/s（1350 續篇）

> 🌐 語言 / Language: [**中文（本頁）/ Chinese**](README.md) · [English](qwen38-5090x2-nccl-mtp-EN.md)

先講結論：這是我 8/26 那篇《雙 5090 跑 Qwen3.8-27B BF16 140K 實測數據與優化心得》（tid 1350）的續篇。那篇的結語是「MTP、NCCL 兩個『理論上應該更快』的方向實測更慢，最後贏的是最樸素的 tensor split + q8_0 KV + 合身 ctx，decode ~48 t/s」。

這一個月我把 NCCL 和 MTP 重新做了一遍，**結論反轉了**：把「NCCL 真正跑起來」這件事做對之後，MTP 的投機解碼收益終於蓋過跨卡同步成本，**decode 從 ~48 t/s 拉到 ~90 t/s，約 1.9×**。重點不是「我編了 NCCL」這麼簡單——1350 那篇已經實測過「編了 NCCL 沒更快」，真正的關鍵是 **NCCL + GPU 間 P2P 同時到位**，這才是當時缺的那一塊。下面把因果鏈和踩的坑寫清楚。

## 1. 硬體與軟體

| 項目 | 配置 |
| --- | --- |
| GPU | 2× RTX 5090 32GB（SM120 Blackwell） |
| CPU | Ryzen 9 9950X3D（16C/32T） |
| 記憶體 | 60GB DDR5 |
| 系統 | Ubuntu 24.04.4 LTS（Kernel 7.0.0-31-generic） |
| 驅動 | 615.71.09（P2P 已開通，見 §4.1） |
| 引擎 | llama.cpp **自編 fork**（build 10702, commit eaf937655，`GGML_CUDA_NCCL=ON`，連 shim NCCL 2.29.7） |
| 模型 | Qwen3.8-27B-BF16（兩片 GGUF 共 54.6GB）＋ mmproj-F16（928MB，啟用 Vision） |
| Context | 95,000（100K 實測 OOM，95K 安全，見 §4.3） |
| KV | q4_0 / q4_0（原 1350 用 q8_0，改 q4_0 省 VRAM 給 MTP，見 §4.4） |
| Split | tensor 0.5,0.5 |
| MTP | **開啟**（`--spec-draft-n-max 5 --spec-draft-n-min 5`，接受率 ~48.5%，見 §3） |
| 用途 | Hermes Agent 主腦，長期單 slot 真實對話流量，非固定短 prompt benchmark |

## 2. 生產啟動參數

以下從運行中進程（PID 6343）的 `/proc/<pid>/cmdline` 直接抓，不是我貼的範例：

```bash
# 關鍵：shim NCCL 必須優先載入（/usr/lib 的 libnccl 在 Blackwell 會卡死，見 §4.1）
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

重點說明：

- `--spec-type draft-mtp`：Qwen3.8-27B 的 MTP draft 層（blk.64）**內嵌在主 GGUF**，不需要 `--model-draft` 另載一個 head。這是跟 Qwen3.8-Flash-Next 不同（那個要另外載 mtp-*.gguf）。
- `--spec-draft-n-max 5`：每輪最多猜 5 個 token。這模型的 draft head 接受率偏低（~48.5%），拉太深會被低接受率拖垮，5 是實測平衡點。
- `-np 1 --kv-unified`：單 slot 長 context。MTP 只在單併發下賺，多併發會反虧（跟 1350 結論一致）。
- `LD_LIBRARY_PATH` 指 shim NCCL 是**強制**的：binary 用 `ldd` 確認連的是 `nccl-shim/lib/libnccl.so.2`，不是 `/usr/lib` 的系統版（原因見 §4.1）。

## 3. 實測數據（生產 log 為準）

以下全部取自 llama-server 自己的 `print_timing` 與 `/metrics`，**不是** client 端量測（client 端 burst 量測會偏高 ~30%，這坑 1350 就提過）。統計區間為本次啟動後真實 Agent 流量：

```text
# /metrics（整體累計）
llamacpp:tokens_predicted_total       28499
llamacpp:tokens_predicted_seconds      314.384  → decode 平均 90.65 tok/s
llamacpp:prompt_tokens_total         330713   （非 cached）
llamacpp:prompt_seconds_total          113.879  → prefill 平均 2904 tok/s
llamacpp:spec_decode_num_accepted_tokens_total  20187
llamacpp:spec_decode_num_draft_tokens_total     41603  → MTP 接受率 48.5%
llamacpp:n_tokens_max               95231    → 單 task 最大 context
llamacpp:requests_deferred               0
```

逐 task 抽樣（每 task 一個 session turn，decode 取該 task 的 `print_timing tg`，prefill 取 `prompt eval time` 終點行）：

```text
task    prompt        prefill     decode (tg)
5998    ~2.0K         2496 t/s    85~94 t/s
7586    ~8.7K         2749 t/s    91~103 t/s
8478    ~5.1K         2476 t/s    95~107 t/s
8848    ~1.4K         1675 t/s    85~100 t/s
5902    ~94.7K *      2967 t/s    （長 context 邊界）
7307    ~36.7K        3410 t/s    ~96 t/s
3675    ~34.1K        2681 t/s    （大 prompt 一次性 prefill）
```

MTP 接受率（`print_timing` 的 `draft acceptance`，23 筆抽樣）：

```text
draft acceptance = 0.53438 (  342 /  640 generated), mean len =  3.67
draft acceptance = 0.76602 (  789 / 1030 generated), mean len =  4.83
draft acceptance = 0.49560 (  451 /  910 generated), mean len =  3.48
draft acceptance = 0.35262 (  573 / 1625 generated), mean len =  2.76
...
23 筆抽樣平均 0.5251，範圍 0.3526 ~ 0.7660
```

觀察：

- **decode 整體 ~90 t/s（`/metrics` 90.65，逐 task `tg` 平均 90.40，範圍 65~113）**，比 1350 那篇的 ~48 t/s 幾乎翻倍。
- **長 context 幾乎不衰减**：94.7K 的 task 5902 prefill 仍跑到 2967 t/s，n_tokens_max 95231，全程 0 OOM、0 deferred。
- **prefill 也跟著上去**：2600~3400 t/s（1350 那篇 ~1158 t/s），NCCL 對 prefill 的 batch 通訊同樣有效。
- **MTP 接受率 48.5%** 偏低（這模型的 draft head 偏弱，比 Qwen3.8-Flash-Next 那篇的 69% 低一截），但 mean len 穩定在 3.3~3.9，配合低同步成本的 NCCL，投機解碼淨收益是正的。
- VRAM 打滿：GPU0 32121 / GPU1 30985 MiB（各 32607 MiB），headroom 只剩 0.5~1.6GB/卡——q4_0 KV + MTP draft 層 + 95K ctx 把空間吃得很緊。

## 4. 實測過的優化方向：這一次哪些真的翻轉了

### 4.1 NCCL——1350 說「編了沒更快」，真正缺的是 P2P

這是本篇核心。**1350 那篇的結論是「雙 5090 走 PCIe 的 internal AllReduce，實測 NCCL 自編版沒有更快，維持現狀」。** 我這次重新做，發現當時的判斷漏了一塊：

- 編 NCCL 只是第一步。Blackwell SM120 上，`/usr/lib` 的系統 libnccl（2.18.3+cud12）init 會**卡死**，根本起不來——所以要用 shim 版 NCCL 2.29.7，`LD_LIBRARY_PATH` 強制讓 binary 連 shim，`ldd` 確認不是 `/usr/lib` 那支。
- **真正讓 NCCL 跑快的，是 GPU 間 P2P 到位**。雙 5090 沒有 NVLink，tensor split 的跨卡 all-reduce 原本走 host 記憶體（PCIe 繞路）。這次驅動升到 615.71.09 + P2P 開通後，`nvidia-smi topo -p2p r` 回 **OK / OK**（之前不是），跨卡通訊走 GPU 直連 P2P，不再繞 host。
- 因果鏈是：**P2P 到位 → 跨卡 all-reduce 成本大降 → MTP 的同步放大陷阱被規避 → MTP 投機解碼的淨收益變正 → decode 翻倍。** 1350 時 MTP 關掉的根因，正是「internal AllReduce 走 PCIe 太慢 + MTP 每輪多次串行 draft forward × 每次跨卡同步」的成本乘法放大。把 P2P 補上之後，這個乘法被拆掉了。

一句話：**「編 NCCL」和「NCCL 跑快」是兩件事，中間隔著一個 P2P。**

### 4.2 MTP——從「實測放棄」到「重開且賺」

1350 時 MTP 關掉的三個原因（接受率 ~40%、佔 VRAM、雙卡同步開銷吃掉收益）裡，**前兩個沒變，變的是第三個**：

- 接受率還是偏低（48.5% vs 1350 那篇記的 ~40%，略有回升但本質還是弱 draft head）；
- MTP draft 層確實佔 VRAM（這也是 KV 從 q8_0 降 q4_0 的原因之一，見 §4.4）；
- 但**跨卡同步成本被 §4.1 的 P2P 拆掉之後，投機解碼從淨負變淨正**，decode 48 → 90 t/s。

驗證 MTP 真在跑的方法（給同樣配置的人）：啟動 log 會有 `creating MTP draft context against the target model`，`/metrics` 的 `spec_decode_num_draft_tokens_total` 持續成長、`print_timing` 有 `draft acceptance = 0.48...` 這種行，三個都齊才是真的在跑。

### 4.3 Context：140K → 95K，因為要給 MTP 讓位

1350 是 140K ctx + q8_0 KV，headroom 約 1~2.3GB/卡。開 MTP 之後 draft 層要吃 VRAM，140K 直接 OOM。實測 **100K OOM、95K 安全**，所以砍到 95K。對 Hermes 這種 context 通常吃不到 100K 的場景，95K 夠用；要更長 context 就得再降 KV 量化或砍 MTP。

### 4.4 KV：q8_0 → q4_0

1350 用 q8_0，這次降 q4_0，原因純粹是 VRAM：MTP draft 層 + 95K ctx 把空間吃滿（GPU0 只剩 0.5GB headroom）。q4_0 對 decode 品質影響很小（KV 量化主要影響長 context 的檢索精度），用 1.5× 的 VRAM 空間換 MTP 能開，是划算的交換。

## 5. 跟 1350 的對照（同一台機、同一個模型）

```text
            1350 (8/26)         本篇 (9/20)
engine      LM Studio 2.29.0    自編 fork build 10702 + shim NCCL 2.29.7
AllReduce   internal (PCIe)     真 NCCL + GPU P2P (topo -p2p r = OK/OK)
MTP         關                  開 (n-max 5, 接受率 48.5%)
KV          q8_0                q4_0
ctx         140K                95K (100K OOM)
decode      ~48 t/s             ~90 t/s  (約 1.9×)
prefill     ~1158 t/s           ~2904 t/s
```

最大的翻轉：**1350 那篇放棄的兩件事（NCCL、MTP），這次因為補上 P2P 這塊拼圖，全部重新變成有效的優化。**

## 6. 給雙卡玩家的參數建議（單 slot 長 context + MTP）

```text
--split-mode tensor --tensor-split 0.5,0.5   # 同型號卡
-fa on                                        # 必開
-ctk q4_0 -ctv q4_0                           # 開 MTP 時 KV 降 q4_0 省 VRAM
-np 1                                         # 單 slot（MTP 只在此下賺）
--ctx-size <合身值>                           # 100K 實測 OOM，95K 安全
-b 4096 -ub 4096
--jinja --metrics
--spec-type draft-mtp --spec-draft-n-max 5    # MTP（接受率偏低別拉太深）
# 前置：NCCL + P2P 雙到位（topo -p2p r = OK/OK）才開 MTP，否則照 1350 關掉
```

## 7. 結語

1350 那篇的「先測再優化」結論是對的，但當時測 NCCL 的樣本漏了 P2P 這塊——**「編了 NCCL 沒更快」的真實原因不是 NCCL 沒用，而是 GPU 間 P2P 沒開通，跨卡通訊還在繞 host 記憶體。** 補上驅動 + P2P 之後，NCCL 才真正發揮，MTP 投機解碼的淨收益跟著變正，decode 從 48 翻倍到 90 t/s。

最大的心得還是那句：**先測再優化**——但測的時候要把變數拆乾淨。我 1350 把「NCCL 沒更快」歸結到 NCCL 本身，這次才發現真正的變數是 P2P。踩過的坑（shim NCCL、P2P、ctx OOM、MTP 接受率偏低）都寫在上面了，希望對同樣在雙卡上想開 MTP 的人有參考價值。

數據全部可重現：引擎端 log（`print_timing`）+ `/metrics`，啟動參數如上。有問題歡迎直接問。

---
*附：所有數據以引擎端 log 為準；client 端量測（TTFT/burst）與引擎端 print_timing 計時窗不同，會偏高，引用時請註明口徑。本篇為 tid 1350 的續篇，硬體/模型/用途不變，差異集中在 §4。*

---
## 技術來源

| 技術/模型 | 官方來源 |
|---|---|
| Qwen3.8-27B（模型） | [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B) |
| llama.cpp（推理框架） | [GitHub](https://github.com/ggml-org/llama.cpp) |
| NCCL（多卡通訊） | [GitHub](https://github.com/NVIDIA/nccl) |
