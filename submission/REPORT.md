# Lab 21 — Evaluation Report

**Họ tên**: Lê Minh Sang  **MSSV**: 2A202602864  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng, Turing sm_75, fp16)`

> Mọi con số dưới đây được đối soát và khớp 100% với các file tạo ra trong `results/`. Phép đo được thực hiện trên tập eval đầy đủ (50 mẫu target, 15 mẫu regression) không qua rút gọn (`smoke_mode: false`).

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (phân chia tất định với seed 42) |
| `max_length` | 1024 — p95 đo được thực tế là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 optimizer steps (cố định trên toàn bộ 4 run huấn luyện) |

**Template có giữ khối `<think>` không?** Có (`keeps_thinking: true`, verdict: `reasoning preserved — safe to train on traces` từ `results/template_check.json`).  
*Xử lý kỹ thuật:* Vì chat template của Qwen3.5 bảo toàn khối suy luận và tự đóng `<think>\n\n</think>\n\n` trong generation prompt đối với các câu trả lời dạng JSON thuần, pipeline `labkit` không dựa vào cờ `assistant_only_loss` mong manh của thư viện mà sử dụng character offsets mapping để cô lập chính xác span của assistant sau `</think>`, đảm bảo loss tính đúng trên output mục tiêu.

---

## 2. Mask proof (NB1)

| Chỉ số kiểm định | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn văn bản giải mã ngược tại vị trí được tính loss (labels $\ne -100$):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3283.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1055.2 |
| (c) LoRA fine-tune | 0.970 | 0.678 | 1.000 | 1372.6 |

**Baseline (b) có thật sự mạnh hơn (a) không?**  
Có, vượt trội hoàn toàn: độ chính xác trường mục tiêu (target) tăng từ 0.000 lên 0.765 (+76.5 điểm %), tỷ lệ đúng định dạng JSON chuẩn (format) tăng từ 0.0% lên 100.0%, và độ trễ giảm từ 3283.5 ms xuống 1055.2 ms do mô hình không còn sinh văn bản lan man ngoài cấu trúc.

**Bạn có sửa `OPTIMIZED_PROMPT` không?**  
Không sửa đổi. Chuỗi hash SHA-256 được bảo toàn tuyệt đối là `719e74d3b6232053`, giữ nguyên độ nghiêm ngặt và tính khách quan của phép thử đối đầu với mô hình fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6266 | 0.970 | 387.6 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5375 | 0.970 | 261.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 387.2 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 455.3 | 3.86 |

> **Quy tắc xếp hạng:** Đánh giá bằng cột **target** ở NB5, không đánh giá bằng train loss ở NB4.

### Phân tích chi tiết 3 câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập kiểm thử target, `attn_only` đạt 0.970, hoà với `correct` (0.970). Tuy nhiên, trên tập huấn luyện, `attn_only` lại có train loss thấp hơn đáng kể (0.5375 so với 0.6266). Thứ tự theo train loss (`attn_only` tốt hơn `correct`) hoàn toàn trái ngược với bản chất tổng quát hoá: việc nhồi rank cực lớn ($r=283$) vào chỉ 2 ma trận Attention ($q, v$) giúp mô hình ghi nhớ (memorize) tập train nhanh hơn, nhưng không tạo ra lợi thế biểu diễn nào trên phân phối kiểm thử so với việc phân bổ đều dung lượng ($r=16$) trên toàn bộ 12 khối Linear của Text Decoder. Kết quả này chứng minh rằng rank không phải là chiếc đũa thần tạo nên chất lượng; vị trí gắn adapter phân bổ rộng khắp các tầng biểu diễn (`all-linear`) mới là nền tảng cốt lõi cho sự ổn định.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Với learning rate ở thang full fine-tune ($1\times 10^{-5}$ thay vì $1\times 10^{-4}$), đường loss suy giảm vô cùng chậm chạp và dường như đi ngang (bắt đầu ở 2.163, kết thúc epoch 2 vẫn dừng ở 1.119, trung bình train loss là 1.5702). Hệ quả là mô hình hoàn toàn thất bại trong việc học cấu trúc phân loại mới: điểm target đạt 0.000 và tỷ lệ format đúng là 0.0%. Nếu một kỹ sư chỉ quan sát đường loss đi xuống từ từ mà không nắm rõ thang độ lớn LR của LoRA, họ sẽ rất dễ đưa ra kết luận sai lầm rằng "mô hình đang hội tụ bình thường nhưng dữ liệu quá khó hoặc cần train thêm hàng chục epoch". Thực tế, cập nhật trọng số trên ma trận low-rank nhân với hệ số co $\alpha / r$ đòi hỏi gradient update bước lớn hơn ~10 lần so với full fine-tuning để phá vỡ các cực tiểu địa phương ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Cấu hình 4-bit `qlora` giảm bộ nhớ VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm đến 56.0% VRAM, cho phép chạy vừa cả trên card đồ họa phổ thông 4GB). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng thêm ~17.5% (455.3 giây so với 387.6 giây do chi phí giải lượng tử hóa dequantization on-the-fly) và điểm số target bị tụt giảm từ 0.970 xuống 0.940 (mất 3.0 điểm % chính xác). Kết quả đo đạc thực tế này hoàn toàn ủng hộ khuyến nghị chính thức của các kỹ sư Unsloth và bài giảng 2026: đối với kiến trúc lai như Qwen3.5, sai số lượng tử hóa tích lũy trên các tầng attention là đáng kể; do đó trên các GPU có đủ VRAM (như Colab T4 16GB có sẵn 14.6GB), lựa chọn tối ưu không hối tiếc luôn là 16-bit LoRA (fp16/bf16).

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết (Phân tích nhân quả):
Cổng hồi quy đưa ra phán quyết `FAILED` bởi vì chỉ số suy thoái năng lực chung vượt quá ngưỡng dung sai cho phép: $\Delta \text{regression} = -0.113$ trong khi ngưỡng chặn trần tối đa là $-0.020$ (suy giảm 11.3% năng lực trả lời câu hỏi phổ thông so với base model). 

Hiện tượng này phản ánh chính xác bài học về **Quên thảm họa (Catastrophic Forgetting)** trong huấn luyện mô hình ngôn ngữ lớn (được phân tích tại deck §6.3). Khi ta fine-tune mô hình chỉ trên 250 mẫu dữ liệu thuần túy về phân loại ticket chăm sóc khách hàng, các gradient updates liên tục kéo các ma trận trọng số thích nghi cực hạn với tác vụ trích xuất thực thể và gán nhãn JSON hẹp. Hậu quả là các không gian biểu diễn ngữ nghĩa rộng phục vụ cho việc suy luận logic và kiến thức tổng quát bị xóa mờ hoặc biến dạng. 

Mặc dù mô hình fine-tune chiến thắng rực rỡ ở tác vụ chuyên môn ($\text{target} = 0.970$ so với baseline (b) $0.765$, tăng +20.5 điểm %), nhưng nó không còn an toàn để triển khai làm một trợ lý hội thoại đa năng. Đây là một kết quả FAILED có giá trị thực tiễn sâu sắc: nó cảnh báo nhóm phát triển không được vội vàng đưa mô hình vào production mà bắt buộc phải áp dụng kỹ thuật **Replay Buffer** — trộn thêm 1% đến 5% dữ liệu chỉ dẫn tổng quát (General Instruction / Alpaca) vào tập huấn luyện để giữ neo các liên kết tri thức nền tảng.

---

## 6. Định tính — Phân tích các ca thắng và ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, đặt chuột không dây VN232232. Trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Sai sentiment (`trung_tinh`) | Đúng 4/4 trường | ✅ **FT thắng**: Fine-tune nhận diện chính xác cảm xúc tích cực từ lời khen cuối câu. |
| 2 | Shop ơi, đặt ốp lưng điện thoại VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Sai urgency (`cao`) | Đúng 4/4 trường | ✅ **FT thắng**: Nhận diện chuẩn mức độ khẩn cấp trung bình dù khách có thái độ bực mình. |
| 3 | Cho mình hỏi, đặt bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Đúng 4/4 trường | Sai urgency (`trung_binh`) | ❌ **FT thua**: Model FT bị bias nhãn `trung_binh`, bỏ qua cụm từ "Khi nào tiện". |
| 4 | Shop ơi, đặt nồi chiên không dầu DH249548. Thiếu phụ kiện. Khi nào tiện. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Đúng 4/4 trường | Sai urgency (`trung_binh`) | ❌ **FT thua**: Lặp lại lỗi thiên kiến độ khẩn cấp đối với cụm từ chỉ thời gian nới lỏng. |
| 5 | Shop ơi, đặt áo khoác gió VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | Đúng 4/4 trường | Sai urgency (`trung_binh`) | ❌ **FT thua**: Mô hình FT không phân biệt được mức độ ưu tiên thấp của khách hàng. |

### Mẫu hình chung ở các ca Fine-tune thua:
Cả 3 ca fine-tune thua (mẫu số 3, 5, 12 trong tập kiểm thử) đều chia sẻ một mẫu số chung rất rõ ràng: **Sự thiên lệch phân phối (distributional bias) đối với trường `urgency`**. Trong tập huấn luyện 225 mẫu, phần lớn các ticket khiếu nại thường mang mức độ khẩn cấp trung bình hoặc cao. Khi gặp các mẫu có tín hiệu ngôn ngữ đặc thù biểu thị mức ưu tiên thấp như *"Khi nào tiện"*, mô hình fine-tune có xu hướng mặc định gán nhãn `urgency: "trung_binh"`. Trong khi đó, Baseline (b) nhờ có phần mô tả schema chi tiết kèm ví dụ few-shot trong ngữ cảnh prompt đã suy luận ngữ nghĩa chính xác hơn trong các trường hợp biên này.

---

## 7. Kết luận & Điều tôi học được

### Kết luận chuyên môn:
Bản checkpoint fine-tune này **chưa nên triển khai trực tiếp ra production như một mô hình đa năng độc lập**, mặc dù nó đã đạt độ chính xác phân loại chuyên biệt rất cao (97.0% trên tập test). Lý do cốt lõi là sự vi phạm cổng hồi quy ($\Delta \text{regression} = -11.3\%$), đồng nghĩa với việc mô hình đã đánh mất một phần khả năng hiểu và tuân thủ các chỉ dẫn ngoài phạm vi hẹp. Tuy nhiên, nếu hệ thống kiến trúc phần mềm tách biệt mô hình này thành một **microservice chuyên biệt (Dedicated Triage Classifier)** chỉ tiếp nhận ticket và trả về JSON định tuyến, mô hình hoàn toàn vượt trội so với giải pháp Prompt Engineering: nó vừa chính xác hơn (+20.5%), vừa triệt tiêu hoàn toàn chi phí token khổng lồ của prompt dài, đồng thời duy trì định dạng chuẩn 100%.

Đòn bẩy quyết định sự thành bại trong lab này không phải là việc tăng rank hay chọn cấu hình LoRA kỳ lạ, mà chính là **Sự kết hợp giữa Loss Mask chuẩn và Thang đo Learning Rate**. Nếu mask bị rò rỉ sang prompt (`everything`), mô hình sẽ học vẹt câu hỏi. Nếu LR đặt sai bậc độ lớn (`wrong_lr`), mô hình tê liệt hoàn toàn. Khi hai yếu tố này chuẩn xác, LoRA 16-bit trên toàn bộ text-linear modules sẽ phát huy sức mạnh tối đa.

### Ba điều tôi học được:
1. **Loss Mask là ranh giới sống còn của SFT**: Không bao giờ được tin tưởng mù quáng vào các cờ mặc định của thư viện cấp cao. Việc kiểm chứng bằng cách giải mã ngược `labels != -100` ra chuỗi ký tự (NB1) là bước bắt buộc để chứng minh tính đúng đắn toán học trước khi tiêu tốn tài nguyên GPU.
2. **Train Loss là chỉ số đánh lừa nguy hiểm nhất**: Mô hình `attn_only` với rank cao ($r=283$) đạt train loss thấp hơn cả `correct` (0.5375 vs 0.6266), nhưng thực chất đó chỉ là hiện tượng ghi nhớ cục bộ trên số ít ma trận. Chỉ có đánh giá trên tập kiểm thử độc lập với các nhóm chỉ số đa chiều (target, format, regression, latency) mới đưa ra phán quyết trung thực.
3. **Cổng hồi quy bảo vệ an toàn hệ thống**: Fine-tuning chuyên biệt luôn tiềm ẩn nguy cơ phá hủy năng lực tổng quát (Catastrophic Forgetting). Một thí nghiệm có kết luận FAILED ở cổng hồi quy mang giá trị kỹ thuật cao gấp nhiều lần một kết quả cherry-pick, bởi nó chỉ ra chính xác điểm yếu cần khắc phục bằng Replay Buffer.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Trộn thêm 3% dữ liệu chỉ dẫn tổng quát tiếng Việt (tương đương 8–10 mẫu đa nhiệm từ tập Alpaca/VieInstruct) vào tập train để kiểm chứng xem cổng hồi quy có lật ngược thành `PASSED` hay không.
2. Thực hiện quét rank có kiểm soát $r \in \{8, 16, 64\}$ trên cấu hình `text-linear` để xác định ngưỡng bão hòa thông tin tối ưu cho tập dữ liệu 250 mẫu này.

---

## Phụ lục — Thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
