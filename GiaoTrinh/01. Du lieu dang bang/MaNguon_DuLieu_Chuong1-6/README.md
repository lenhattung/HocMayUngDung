# Kỹ thuật Học máy cho Dữ liệu Bảng – Tập 1
## Mã nguồn và dữ liệu Chương 1–6

```
MaNguon_DuLieu_Chuong1-6/
├── data/
│   ├── credit_default.xls     # Default of Credit Card Clients (Yeh & Lien, 2009) – 30 000 mẫu
│   └── cal_housing.data       # California Housing (Pace & Barry, 1997) – 20 640 mẫu
├── code/
│   ├── common.py              # hàm dùng chung: nạp dữ liệu, chia train/test, mô hình, chỉ số
│   ├── run_chapter.py         # chạy lần lượt các ô lệnh của một chương
│   ├── ch01/ … ch06/          # mỗi file sNN_*.py là một "Ô lệnh" trong sách
├── notebooks/                 # ch01_notebook.ipynb … ch06_notebook.ipynb (đúng các ô lệnh trong sách)
├── figures/                   # hình sinh ra khi chạy code
└── requirements.txt
```

### Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
(Chương 6 cần PyTorch; bản CPU là đủ.)

### Cách 1 – Chạy bằng notebook (khuyên dùng khi học)
Chép notebook của chương cần học **ra thư mục gốc** (cùng cấp với `data/` và `code/`),
mở bằng Jupyter/VS Code rồi chạy lần lượt từng ô.
Các ô lệnh trong một chương dùng chung biến, nên phải chạy theo đúng thứ tự.

### Cách 2 – Chạy cả chương bằng script
Đứng ở thư mục gốc:
```bash
python code/run_chapter.py ch01
```
Kết quả in ra được ghi vào `outputs/ch01/`, hình vào `figures/`.

### Lưu ý
- Chia train/test cố định: 80/20, phân tầng, `random_state=42` – để kết quả giữa các chương so sánh được.
- Con số có thể lệch nhỏ ở chữ số cuối so với sách do khác phiên bản thư viện
  (sách dùng scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, imbalanced-learn 0.14, PyTorch 2.14).
- Ô lệnh 8 Chương 6 huấn luyện 40 mạng nơ-ron, mất khoảng 2–3 phút trên CPU.
