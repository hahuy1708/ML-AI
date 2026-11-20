# Dữ liệu và sản phẩm kiểm toán

## 1. Tổng quan

- **Nguồn dữ liệu:** Kaggle, `techsash/waste-classification-data`.
- **Bài toán:** phân loại nhị phân rác hữu cơ và rác tái chế.
- **Nhãn:** `O` (organic) mã hóa là `0`; `R` (recyclable) mã hóa là `1`. Lớp dương (positive class) là `R`.
- **Quy mô:** 25.077 tệp ảnh nằm trong hai thư mục chính thức `TRAIN` và `TEST`.
- **Bản sao:** tải về từ Kaggle có hai bản thư mục. Kiểm toán xác nhận hai bản giống hệt nhau về MD5; ảnh chỉ được phân tích trên một bản.
- **Dữ liệu ảnh:** repository này không lưu tệp ảnh. Người dùng cần tải hoặc đính kèm (attach) dataset từ Kaggle.

| Split chính thức | Organic (`O`) | Recyclable (`R`) | Tổng | Tỉ lệ `R` |
|---|---:|---:|---:|---:|
| `TRAIN` | 12.565 | 9.999 | 22.564 | 44,31% |
| `TEST` | 1.401 | 1.112 | 2.513 | 44,25% |

## 2. Đặc điểm ảnh

- **Định dạng:** toàn bộ 25.077 tệp có phần mở rộng `.jpg`, nhưng kiểm tra nội dung cho thấy 24.705 ảnh JPEG và 372 ảnh PNG. Các ảnh PNG ở chế độ bảng màu (`P`). Khi đưa ảnh vào mô hình, cần chuyển về RGB.
- **Kích thước:** chiều rộng từ 51 đến 1.460 px (trung vị 259 px); chiều cao từ 69 đến 1.500 px (trung vị 194 px).
- **Ảnh nhỏ:** 19.767 ảnh (78,8%) có ít nhất một cạnh nhỏ hơn 224 px. Các ảnh này sẽ được phóng to khi đưa vào mô hình 224 × 224.
- **Tính toàn vẹn:** không phát hiện ảnh lỗi hoặc không đọc được trong lần kiểm toán này.

## 3. Các tệp trong thư mục này

| Tệp | Nội dung | Mục đích | Hạn chế hiện tại |
|---|---|---|---|
| `splits.csv` | Mỗi ảnh một dòng gồm `path`, `label`, `split`, `group_id`, `test_leak` | Phân chia train/validation/test dùng chung cho các thí nghiệm | Cột `path` là đường dẫn tuyệt đối trên Kaggle, nên tệp này chưa dùng được trên máy khác. Phân chia còn ở trạng thái tạm thời (xem mục 5). |
| `manifest.csv` | Thông tin từng ảnh: split chính thức, nhãn, phần mở rộng, MD5, dHash, kích thước, chế độ và định dạng ảnh, trạng thái đọc | Kiểm kê dữ liệu, phân tích khám phá (EDA), tìm ảnh lỗi và ảnh trùng | Đường dẫn tuyệt đối. Đây là sản phẩm trung gian do script sinh ra. |
| `near_dup_pairs.csv` | Các cặp ảnh thuộc hai split chính thức khác nhau có dHash gần nhau, kèm khoảng cách Hamming và nhãn | Kiểm tra các cặp nghi ngờ trùng giữa train và test | Đây là ứng viên, chưa phải trùng lặp ngữ nghĩa đã xác nhận. Đường dẫn tuyệt đối. |
| `audit_report.json` | Thống kê tổng hợp, kết quả nhóm trùng lặp, số lượng theo split, thời gian chạy, và checksum của `splits.csv` | Ghi nhận và so sánh các lần kiểm toán | Chứa đường dẫn tuyệt đối. Mô tả đúng lần chạy này, không phải bản phát hành dữ liệu cuối cùng. |

Script tạo ra các tệp trên là `scripts/00_dataset_audit.py`.

### 3.1. Ý nghĩa các cột của `splits.csv`

- **`path`:** đường dẫn đến ảnh gốc. Trong tệp hiện tại, đường dẫn là tuyệt đối (`/kaggle/input/...`). Cần chuyển thành đường dẫn tương đối so với thư mục gốc của dataset trước khi dùng làm bảng chia chung của nhóm.
- **`label`:** `0` với organic, `1` với recyclable.
- **`split`:** một trong các giá trị `train`, `val`, `test`. Tập `test` là thư mục `TEST` chính thức và không bị thay đổi.
- **`group_id`:** định danh của thành phần liên thông, được hình thành từ các cặp trùng MD5 hoàn toàn và các cặp dHash có khoảng cách không vượt ngưỡng cấu hình. Các ảnh cùng `group_id` luôn nằm trong cùng một split.
- **`test_leak`:** nhận giá trị `1` nếu ảnh thuộc `test` có ảnh trùng hoặc gần trùng trong tập train chính thức, và `0` trong các trường hợp còn lại. Với ngưỡng dHash hiện tại, cờ này chỉ là tín hiệu cần rà soát, chưa chứng minh hai ảnh thể hiện cùng một đối tượng.

## 4. Kết quả kiểm toán hiện tại

Kết quả dưới đây thu được với `near_dist = 4`, `seed = 42`, `val_frac = 0,15` và 4 tiến trình song song.

| Split trong `splits.csv` | Organic (`0`) | Recyclable (`1`) | Tổng |
|---|---:|---:|---:|
| `train` | 10.829 | 8.712 | 19.541 |
| `val` | 1.736 | 1.287 | 3.023 |
| `test` | 1.401 | 1.112 | 2.513 |

Tập validation chiếm 3.023 trên 22.564 ảnh train chính thức, tức 13,4%, thay vì đúng 15%. Nguyên nhân là việc phân chia được thực hiện theo nhóm, nên không thể đảm bảo đúng tỉ lệ từng ảnh.

Các phát hiện chính:

- **Trùng lặp MD5 hoàn toàn:** 326 nhóm, tương ứng 326 bản sao thừa.
- **Cặp gần trùng dHash:** 41.338 cặp có khoảng cách Hamming không vượt 4, trong đó 6.422 cặp có khoảng cách 0.
- **Nhóm lớn nhất:** 1.924 ảnh. Có 76 nhóm chứa cả hai nhãn. Vì nhóm được hình thành theo thành phần liên thông (phép bắc cầu), một chuỗi các cặp gần nhau từng cặp có thể tạo thành một nhóm lớn dù hai đầu chuỗi khác nhau đáng kể.
- **Ảnh test có ảnh trùng trong train:** 552 trên 2.513 ảnh test (22,0%) được gắn cờ theo quy tắc MD5 và dHash hiện tại. dHash có thể gắn cờ các ảnh khác nhau nhưng ít chi tiết, nên tỉ lệ này cần được kiểm tra bằng mắt và bằng MD5 trước khi mô tả là rò rỉ dữ liệu đã xác nhận.
- **Thời gian chạy:** 364,5 giây với 4 tiến trình, ghi trong `audit_report.json`.

## 5. Trạng thái và hạn chế

**Trạng thái: tạm thời.** Không dùng `splits.csv` hiện tại làm bảng chia cuối cùng cho các thí nghiệm nhóm cho đến khi:

1. Đường dẫn được chuyển thành tương đối.
2. Các nhóm lớn (đặc biệt nhóm 1.924 ảnh) và các nhóm chứa cả hai nhãn được kiểm tra bằng mắt.
3. Cờ `test_leak` được rà soát, tách riêng các trường hợp trùng MD5 khỏi các trường hợp chỉ gần trùng theo dHash.
4. Quy tắc gộp nhóm (ngưỡng khoảng cách 4 hoặc 0) được quyết định và ghi lại.

Các con số trong tài liệu này mô tả một lần kiểm toán cụ thể. Chúng không phải là thông số chính thức của dataset.

## 6. Tái lập kết quả

Chạy từ thư mục gốc của dự án `waste-cnn`, khi dataset đã được đính kèm hoặc tải về cục bộ:

```bash
python scripts/00_dataset_audit.py \
  --root <thư mục chứa các thư mục DATASET có cấu trúc TRAIN/ và TEST/> \
  --out dataset/ \
  --workers 4
```

Script yêu cầu cấu trúc thư mục: mỗi thư mục gốc chứa `TRAIN/` và `TEST/`, và mỗi thư mục này có các thư mục lớp `O/` và `R/`. Script ghi đè bốn sản phẩm nêu trong mục 3, nên cần kiểm tra lại đường dẫn `--root` và `--out` trước khi chạy.

## 7. Kiểm tra toàn vẹn

MD5 của `splits.csv` hiện tại: `66c0bcf5ba56ddfe7bc0ba4d5246e7a8`.

Giá trị này chỉ áp dụng cho tệp tạm thời hiện có. Khi tệp được tạo lại hoặc đường dẫn được chuyển thành tương đối, giá trị sẽ thay đổi và cần cập nhật tại đây.