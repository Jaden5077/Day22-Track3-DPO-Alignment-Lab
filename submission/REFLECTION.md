# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Nguyễn Đức Dũng_
**Cohort:** _A20-K1_
**Tier đã chạy:** _<T4>_
**Date:** _<2026-05-08>_

---

## 1. Setup

| Item                     | Value                                                              |
| ------------------------ | ------------------------------------------------------------------ |
| GPU                      | _Free Colab T4 16GB_                                               |
| CUDA / driver            | Build cuda_12.8.r12.8/compiler.35583870_0                          |
| Base model               | _unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit_                           |
| SFT dataset slice        | _5CD-AI/Vietnamese-alpaca-gpt4-gg-translated · 1000 pairs_         |
| Preference dataset slice | _argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs_ |
| `COMPUTE_TIER` env       | T4                                                                 |
| Total cost               | $0 (free Colab)                                                    |

---

## 2. DPO experiment results

| Metric                                          |    SFT-only baseline |                  SFT + DPO |
| ----------------------------------------------- | -------------------: | -------------------------: |
| Training time (NB3)                             |                    — |           _<e.g., 28 min>_ |
| VRAM peak                                       |    _<e.g., 10.4 GB>_ |          _<e.g., 13.8 GB>_ |
| Final loss                                      | _<e.g., 1.82 (SFT)>_ |       _<e.g., 0.48 (DPO)>_ |
| Reward gap (chosen − rejected, end of training) |                  n/a |             _<e.g., 1.34>_ |
| Mean output length                              | _<e.g., 142 tokens>_ | _<e.g., 87 tokens (-39%)>_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):

- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste `03_dpo_reward_curves.png` here** (or link to it in `submission/screenshots/`).

_Interpret both `chosen_rewards` and `rejected_rewards` separately. Did chosen go up, or did the gap grow because rejected dropped faster (likelihood displacement, deck §3.4)? What does this tell you about whether DPO did what you wanted? Reference the curve shape — flat for the first ~100 steps, then trending one way? KL divergence to reference at end?_

_Answer here. ≥ 100 words._

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| #   | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner                |
| --- | --------------- | ------------------ | -------- | ------- | --------------------- |
| 1   | helpfulness     | _<...>_            | _<...>_  | _<...>_ | _<SFT \| DPO \| tie>_ |
| 2   | helpfulness     |                    |          |         |                       |
| 3   | helpfulness     |                    |          |         |                       |
| 4   | helpfulness     |                    |          |         |                       |
| 5   | safety          |                    |          |         |                       |
| 6   | safety          |                    |          |         |                       |
| 7   | safety          |                    |          |         |                       |
| 8   | safety          |                    |          |         |                       |

**Win/loss/tie summary:** _<e.g., SFT+DPO wins 5/8, ties 2/8, loses 1/8>_

**Judge used:** _<gpt-4o-mini | claude-haiku-4-5 | manual rubric>_

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

|             β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
| ------------: | ---------: | -------------------: | ------------: | ----- |
|          0.05 |    _<...>_ |              _<...>_ |       _<...>_ |       |
| 0.1 (default) |    _<...>_ |              _<...>_ |       _<...>_ |       |
|           0.5 |    _<...>_ |              _<...>_ |       _<...>_ |       |

_Interpret: where's the sweet spot for your data? Why? Does it match the deck's §3.3 prediction?_

_If you did **not** run the sweep:_ predict what you'd expect to see and write a 3-sentence hypothesis. (No points lost — but the muscle of forming a hypothesis is the value.)

_Answer here._

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

Quyết định quan trọng nhất tôi đã thực hiện trong lab này là chuyển đổi từ mô hình mặc định Qwen2.5-3B sang Qwen2.5-0.5B (cùng với việc sử dụng tier T4 thay vì BigGPU).
1. Giải pháp thay thế mà tôi cân nhắc ban đầu là giữ nguyên cấu hình mô hình 3B.
2. Tôi chọn 0.5B vì nhận thấy việc huấn luyện một mô hình 3B trên Colab T4 miễn phí mất rất nhiều thời gian. Việc giảm kích thước mô hình xuống 0.5B giúp rút ngắn đáng kể thời gian training SFT và DPO, tối ưu hóa tài nguyên phần cứng giới hạn.
3. Ban đầu kết quả diễn ra đúng như dự kiến, quá trình training DPO diễn ra rất nhanh gọn. Tuy nhiên, điều bất ngờ và đáng tiếc là dù đã cố gắng tiết kiệm thời gian huấn luyện, tôi vẫn dính lỗi timeout ở bước export GGUF. Dù tôi đã tìm đủ mọi cách để ép thư viện dùng bản pre-built, lệnh `model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q4_k_m")` vẫn nằng nặc ngầm kích hoạt việc build `llama.cpp` từ mã nguồn (source). Quá trình này tốn mất cả tiếng đồng hồ và làm cạn kiệt hoàn toàn thời gian sử dụng T4 free tier, khiến các bước cuối của lab thất bại.
4. Nếu làm lại lab vào ngày mai, tôi vẫn sẽ chọn mô hình 0.5B để tiết kiệm thời gian. Tuy nhiên, do không thể ngăn được sự cứng đầu của thư viện trong việc tự build từ source, tôi sẽ quyết định bỏ qua hoàn toàn bước xuất GGUF. Việc dứt khoát bỏ qua này sẽ giúp tôi tiết kiệm tài nguyên GPU quý giá để ưu tiên chạy xong trọn vẹn bước đánh giá (benchmark) cuối cùng.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark       | SFT-only | SFT+DPO |       Δ |
| --------------- | -------: | ------: | ------: |
| IFEval          |  n/a | n/a | n/a |
| GSM8K           |  n/a | n/a | n/a |
| MMLU (sampled)  |  n/a | n/a | n/a |
| AlpacaEval-lite |  n/a | n/a | n/a |

Do sự cố timeout của Colab ở bước export GGUF (như trình bày ở phần 6), tôi không thể chạy tiếp các bước LLM benchmark. Quá trình build `llama.cpp` từ source do lệnh `model.save_pretrained_gguf` gọi ngầm mất quá nhiều thời gian làm cạn giới hạn T4 GPU free tier. Vì vậy, tôi chưa thể thu thập và so sánh điểm số giữa mô hình SFT-only và SFT+DPO cho lab này.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

Điều đáng ngạc nhiên nhất là dù tôi đã thử tìm mọi cách cấu hình dùng bản pre-built, một hàm tưởng chừng rất tiện lợi như `model.save_pretrained_gguf(..., quantization_method="q4_k_m")` vẫn nằng nặc tự build toàn bộ mã nguồn của `llama.cpp` trong môi trường Colab. Sự cứng đầu của thư viện này gây mất cả tiếng đồng hồ và trực tiếp làm cạn kiệt tài nguyên GPU T4 của free tier, khiến toàn bộ các bước đánh giá (benchmark) cuối lab thất bại ngoài dự kiến.
