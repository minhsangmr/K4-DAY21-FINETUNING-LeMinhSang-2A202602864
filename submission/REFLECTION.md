# Reflection — Lab 21

*Họ tên: Lê Minh Sang — MSSV: 2A202602864*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng ở run `attn_only`: khi nâng rank lên $r=283$ để khớp đúng tổng số tham số trainable với `correct`, train loss của `attn_only` thậm chí còn giảm sâu hơn cấu hình chuẩn (0.5375 so với 0.6266). Nếu nhìn vào biểu đồ loss, người ta sẽ tưởng `attn_only` thắng cuộc, nhưng khi đo trên tập test độc lập thì cả hai chỉ hòa nhau ở 0.970. Điều này chứng minh trực quan việc nhồi rank vào ít ma trận chỉ làm mô hình ghi nhớ tập train nhanh hơn chứ không gia tăng năng lực biểu diễn tổng quát.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở công đoạn sinh văn bản đánh giá (inference / greedy decode) ở NB2 và NB5 trên GPU T4 (tổng cộng mất gần 40 phút), chứ không phải ở bước huấn luyện NB3 (chỉ mất ~6.5 phút). Điều này nằm ngoài dự đoán ban đầu vì thông thường người mới học nghĩ rằng huấn luyện mới là phần tốn kém nhất; thực tế việc sinh autoregressive tuần tự cho 50 mẫu đánh giá qua 3 mốc baseline ngốn thời gian áp đảo.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng "chỉ cần train loss giảm đều và validation loss đẹp thì mô hình đã thành công". Bây giờ tôi hiểu rằng train loss có thể là một ảo ảnh nguy hiểm. Thêm vào đó, việc fine-tune chuyên biệt cực kỳ dễ gây ra Catastrophic Forgetting (suy giảm 11.3% năng lực tổng quát ở cổng hồi quy). Một mô hình đạt 97% task chuyên môn vẫn có thể thất bại trong thực tế nếu không có cổng kiểm thử hồi quy bảo vệ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi sử dụng AI assistant để thiết lập môi trường ảo bằng `uv`, đối chiếu các lỗi phiên bản trong stack phần mềm (như tương thích `torchao` và chat template Jinja2), và tự động hóa kiểm định các kết quả đo đạc từ các file JSON/CSV. Điểm AI assistant dễ bị nhầm nếu không giám sát là việc tự động kết luận "kết quả FAILED ở cổng hồi quy là thí nghiệm hỏng" — trong khi theo triết lý của lab, việc phát hiện và chứng minh được mô hình bị hồi quy năng lực là một kết quả nghiên cứu trung thực và đạt điểm tối đa.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở code hay thuê GPU để train, mà là: **Thiết lập Baseline (b) bằng Prompt Engineering tối ưu kèm bộ kiểm định đa chiều (Target, Format, Regression Gate)**. Nếu một prompt được thiết kế tử tế đã giải quyết được 80–90% bài toán với chi phí và độ phức tạp thấp, tôi sẽ cân nhắc kỹ trước khi fine-tune. Nếu bắt buộc fine-tune, tôi sẽ kiểm chứng Loss Mask bằng mắt và giải mã ngược trước khi bắt đầu huấn luyện.
