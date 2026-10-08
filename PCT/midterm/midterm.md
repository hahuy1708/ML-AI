# MIDTERM.md

> **Đọc file này TRƯỚC KHI làm bất cứ việc gì.** 

## 1. Input - Output

### **Input:**

- **Đề bài:** `midterm/debai.docx` 
- **Dữ liệu CSV:** `midterm/data/` (`car_crashes`, `diamonds`, `mpg`, `planets`, `taxis`, `wine`).
- **Ảnh:** tập CBSD68, nằm trong `midterm/data/CBSD68/` (tìm bằng `glob` nếu chưa rõ đường dẫn).
- **Repo tham khảo (chỉ đọc, không sửa):**
  - `01-pca_lda/` → `PCA.ipynb`, `LDA.ipynb`
  - `02-clustering/fuzzy-c-means/` → `c_means_clustering.ipynb`, `fcm_image_segmentation.ipynb`
  - `02-clustering/k-means/` → `k_means_clustering.ipynb`, `kmeans_image_segmentation.ipynb`
  - `03-knn/` → `knn_A_B.ipynb`, `knn_C.py`

### **Output**
 
**Một notebook duy nhất:** `midterm/Midterm.ipynb`, gồm cả 3 câu theo thứ tự Câu 1 → Câu 2 → Câu 3. Mỗi câu bắt đầu bằng 1 ô markdown ghi tiêu đề câu.

## 2. Rules
 
- **Dùng kernel / venv có sẵn của repo** : PCT\venv\Scripts\python.exe 
    - Không tạo kernel mới, không tạo venv mới.
    - Các thư viện cần thiết đã có trong venv. Chỉ khi `import` lỗi mới báo lại cho người dùng, không tự `pip install`.
- **Đọc notebook tham khảo trong repo trước khi viết code**, để bám theo cách trình bày (cấu trúc, đặt tên hàm, kiểu biểu đồ, cách viết nhận xét) và tái sử dụng code đã có.
- Chỉ tạo/sửa `midterm/Midterm.ipynb`. Không đụng vào file khác trong repo.
- Phần "tự cài" chỉ dùng NumPy, Matplotlib (và `time`), viết **theo từng bước**, mỗi bước một hàm, vector hóa, không lặp từng pixel/điểm.
- Cố định seed (`42`). Mỗi biểu đồ có tiêu đề và nhãn trục.
- Markdown trong notebook ngắn gọn (tối đa 3 dòng/ô). Mỗi câu kết thúc bằng phần **Nhận xét** 3–5 gạch đầu dòng.
- Không làm thêm ngoài đề, không tinh chỉnh siêu tham số mất thời gian.
- Làm xong: **Restart Kernel → Run All**, không còn lỗi. Báo lại trong chat tối đa 5 dòng (đã làm gì, giả định nào).


## 3. Quy trình chung cho mỗi notebook

Cấu trúc ô cố định:

1. Markdown: tiêu đề câu + yêu cầu đề (chép lại 1–2 dòng).
2. Code: import + seed.
3. Code: đọc dữ liệu + in `shape`, `head()`, `dtypes`, số giá trị thiếu.
4. Code: tiền xử lý.
5. Code/Markdown: phần chính theo từng bước.
6. Code: trực quan.
7. Markdown: **Nhận xét kết quả (3–5 gạch đầu dòng).**

## 4. Câu 1 — PCA (3 điểm)

**Đề:** Cho tập dữ liệu `*.csv`, giảm chiều bằng PCA và trực quan kết quả. **Số chiều đầu vào ≥ 7, số chiều đầu ra ≤ 4.**

Các bước:

1. **Chọn dữ liệu:** dùng file CSV đề chỉ định; nếu đề không chỉ định, chọn file trong `midterm/data/` có **≥ 7 cột số**. Kiểm tra bằng code (`df.select_dtypes("number").shape[1]`), không đoán. Gợi ý ứng viên thường có đủ cột số: `car_crashes`, `mpg`, `diamonds`, `wine` (cần xác nhận bằng code).
2. **Tiền xử lý:** chỉ giữ cột số; xử lý giá trị thiếu (`dropna()` hoặc điền median, ghi rõ chọn cách nào); **chuẩn hóa bằng `StandardScaler`** trước PCA. In ra số chiều đầu vào (phải ≥ 7).
3. **PCA:** `PCA(n_components=...)` với n_components ≤ 4 (chọn 3 hoặc 2; nên chọn sao cho đạt được cả khả năng vẽ). In ra: `explained_variance_ratio_`, tỉ lệ cộng dồn.
4. **Trực quan (tối thiểu):**
   - Biểu đồ phương sai giải thích (bar + đường cộng dồn).
   - Scatter 2D theo PC1–PC2; tô màu theo 1 cột phân loại nếu dữ liệu có (ví dụ `origin`, `cut`, `class`...). Nếu giữ 3 thành phần, thêm scatter 3D.
5. **Nhận xét:** số chiều giảm từ N → k, giữ được bao nhiêu % phương sai, cột nào đóng góp lớn nhất vào PC1 (xem `components_`).

## 5. Câu 2 — Fuzzy C-Means (3 điểm)

**Đề:** Phân cụm bằng FCM, **số cụm m ≥ 5**, trực quan kết quả. **Tự cài từng bước (không thư viện) và có dùng thư viện, so sánh kết quả.**

> ⚠️ Trong đề, `m` là **số cụm**. Trong công thức FCM, hệ số mờ (fuzzifier) thường cũng ký hiệu `m`. Để tránh nhầm: đặt `n_clusters = 5` (hoặc lớn hơn) và `fuzzifier = 2.0`.

**Dữ liệu:** dùng cùng dữ liệu Câu 1 sau khi chuẩn hóa (hoặc dữ liệu đề cho). Trực quan trên chiều đã giảm bằng PCA 2D.

### 5.1 Tự cài (không thư viện) — chia hàm theo bước

Dùng tiêu chuẩn: hàm mục tiêu, tâm cụm, ma trận thành viên.

- Bước 1 — `init_membership(n_samples, n_clusters)`: ma trận `U` ngẫu nhiên, mỗi hàng cộng bằng 1.
- Bước 2 — `update_centers(X, U, fuzzifier)`: `v_i = Σ_j u_ij^p · x_j / Σ_j u_ij^p`
- Bước 3 — `update_membership(X, V, fuzzifier)`: `u_ij = 1 / Σ_k (d_ij / d_kj)^(2/(p−1))`, trong đó `d_ij = ||x_j − v_i||`. **Cộng epsilon nhỏ (1e-10) vào khoảng cách để tránh chia cho 0.**
- Bước 4 — `objective(X, U, V, fuzzifier)`: `J = Σ_i Σ_j u_ij^p · d_ij²`
- Bước 5 — vòng lặp: lặp bước 2–3 đến khi `||U_new − U_old|| < 1e-5` hoặc đạt `max_iter = 150`. Lưu `J` theo từng vòng.
- Bước 6 — gán nhãn cứng: `labels = argmax(U, axis=1)`.

### 5.2 Dùng thư viện

- Dùng `skfuzzy.cluster.cmeans`. Lưu ý `skfuzzy` nhận dữ liệu dạng **(n_features, n_samples)** và trả về `u` dạng **(n_clusters, n_samples)** → cần chuyển vị cho khớp.
- **Truyền cùng ma trận khởi tạo `U` ban đầu** (tham số `init`) như bản tự cài để so sánh công bằng.

### 5.3 So sánh (bắt buộc có bảng)

| Tiêu chí | Tự cài | Thư viện |
|---|---|---|
| Số vòng lặp hội tụ | | |
| Giá trị hàm mục tiêu J cuối | | |
| FPC (Fuzzy Partition Coefficient) | | |
| Thời gian chạy (s) | | |
| Độ tương đồng nhãn (ARI hoặc % điểm cùng nhãn sau khi ghép tâm cụm) | | |

Trực quan: hai scatter cạnh nhau (tự cài vs thư viện) trên PCA 2D, đánh dấu tâm cụm; thêm đường cong J theo vòng lặp của bản tự cài.

Nhận xét: hai kết quả có khớp không, khác ở đâu, vì sao (khởi tạo, ngưỡng dừng, thứ tự cụm có thể bị hoán đổi nên cần ghép cụm trước khi so sánh).

## 6. Câu 3 — Phân đoạn ảnh màu (4 điểm)

**Đề:** Chọn **01 ảnh màu** bất kỳ từ tập **CBSD68**. Phân đoạn ảnh màu với **số nhóm ≥ 3**. **Tự cài từng bước + dùng thư viện, so sánh.**

**Chọn thuật toán:**
- Nếu đề nêu rõ thuật toán → dùng đúng thuật toán đó.
- Nếu đề không nêu → dùng **FCM** (tái sử dụng hàm từ Câu 2; tham khảo `fcm_image_segmentation.ipynb`). Nếu còn dư thời gian, mới thêm K-Means như phần bổ sung.

Các bước:

1. Đọc 1 ảnh CBSD68 (RGB), hiển thị ảnh gốc, in kích thước. Ghi rõ tên file ảnh đã chọn.
2. Chuẩn hóa pixel về `[0, 1]`, reshape `(H*W, 3)`. Giữ nguyên kích thước ảnh; **chỉ resize nhỏ lại nếu bản tự cài chạy quá chậm**, và phải ghi chú trong notebook.
3. `n_clusters = 4` (hoặc 3; **không được < 3**).
4. **Tự cài** theo đúng các hàm bước như Câu 2 (init → update centers → update membership → kiểm tra hội tụ → gán nhãn). Dữ liệu là pixel RGB.
5. **Thư viện:** `skfuzzy.cmeans` (nếu FCM) hoặc `sklearn.cluster.KMeans` (nếu K-Means), cùng số cụm và cùng khởi tạo nếu có thể.
6. **Dựng lại ảnh phân đoạn:** thay mỗi pixel bằng màu tâm cụm của nó, `reshape` lại `(H, W, 3)`.
7. **Trực quan:** 1 hàng 3 ảnh: Gốc | Tự cài | Thư viện. Thêm hiển thị màu tâm cụm (palette) nếu còn thời gian.
8. **So sánh (bảng):** thời gian chạy, số vòng lặp, hàm mục tiêu J (hoặc inertia), độ tương đồng nhãn giữa hai bản (ARI hoặc % pixel trùng nhãn sau ghép cụm), nhận xét trực quan.
