```markdown
📘 Readee — AI Reading Assistant

"Learning should adapt to you — not the other way around."

Readee là dự án R&D của bộ phận AI tại VOCA, hướng tới việc xây dựng một nền tảng học tiếng Anh thông minh, cá nhân hóa dựa trên dữ liệu — giải quyết vấn đề thiếu tài liệu đọc phù hợp trình độ (CEFR) và sở thích.

🎯 Tầm nhìn & Sứ mệnh

Tầm nhìn: Xây dựng AI-driven English Content Engine — cỗ máy tạo nội dung học tiếng Anh theo yêu cầu.
Sứ mệnh: Ứng dụng triết lý Comprehensible Input để biến việc học ngôn ngữ trở nên tự nhiên và hiệu quả.

🚀 Tính năng Cốt lõi
1. 🧠 AI Phân loại (Classification Engine)
Phân loại bài đọc theo chuẩn CEFR (A1–B2).
Công nghệ: XGBoost + Word2Vec
Ứng dụng: Tự động đánh giá độ khó văn bản.

2. ✍️ AI Tạo sinh (Generative Engine)
Sinh truyện ngắn/hội thoại theo trình độ + sở thích.
Đầu ra tối ưu cho cá nhân hóa nội dung.

3. 🎓 Hệ thống Bổ trợ (Interactive Learning)
Quiz Generator
Smart Flashcards

🛠️ Tech Stack
| Hạng mục | Công nghệ | Mục đích |
| :--- | :--- | :--- |
| Ngôn ngữ lập trình | Python | Xây dựng mô hình & ứng dụng |
| Xử lý dữ liệu | Pandas, NumPy, NLTK | Cleaning, Tokenization, Lemmatization |
| Feature Engineering | Word2Vec (Gensim) | Biểu diễn ngữ nghĩa |
| Mô hình AI | XGBoost, Scikit-learn | Phân loại CEFR |
| Cân bằng dữ liệu | SMOTE | Giảm mất cân bằng lớp |
| XAI | SHAP | Giải thích mô hình |
| App Demo | Streamlit | Giao diện thử nghiệm |

📈 Kết quả Kỹ thuật
Dataset chuẩn hóa: 1.235 mẫu (A1–B2).
Accuracy mô hình (XGBoost + Word2Vec + SMOTE): ~76%.
Recall lớp khó B1: 71% (cải thiện lớn).
Tốc độ dự đoán: < 1 giây/request.

## 👥 Đội ngũ Phát triển (Group 5 – Readee Team)

| # | Tên thành viên | Vai trò | Vai trò trong dự án |
| :---: | :--- | :---: | :--- |
| 1 | **Dương Minh Kha** | ![Leader](https://img.shields.io/badge/-Leader-red) | Điều phối; Technical Architect; Review chất lượng sản phẩm |
| 2 | **Đỗ Liên Thịnh** | ![Member](https://img.shields.io/badge/-Member-success) | Data Engineer: Thu thập, làm sạch & chuẩn hóa dữ liệu |
| 3 | **Huỳnh Đằng Phát** | ![Member](https://img.shields.io/badge/-Member-success) | AI Engineer: Training mô hình CEFR, Embedding, Classification |
| 4 | **Lê Thanh Hằng** | ![Member](https://img.shields.io/badge/-Member-success) | Frontend App, UI/UX, giao diện Notion App |
| 5 | **Trần Thị Khánh Linh** | ![Member](https://img.shields.io/badge/-Member-success) | Business Analyst: Use-case, user flow, giá trị sản phẩm |
| 6 | **Bùi Lê Đức Đạt** | ![Member](https://img.shields.io/badge/-Member-success) | AI Pipeline + Evaluation: SHAP, đánh giá mô hình |
| 7 | **Nguyễn Tuyết Minh** | ![Member](https://img.shields.io/badge/-Member-success) | Frontend App, UI/UX, giao diện Notion App |


📂 Cấu trúc Dự án
```text
Readee-Project/
├── app/
│   ├── app.py                 # Ứng dụng Demo Streamlit
│   ├── models/                # Model (.pkl), vectorizer
│   └── utils/                 # Hàm xử lý
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_Baseline_Model.ipynb
│   └── 4_Final_Model.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── docs/                      # Báo cáo, Slide, AI Canvas
├── requirements.txt
└── README.md


🏃 Hướng dẫn Chạy Demo
1. Clone Project
git clone https://github.com/phatdang2403-hub/Readee-Project.git
cd Readee-Project

2. Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate


Hoặc Mac/Linux:

source venv/bin/activate

3. Cài thư viện
pip install -r requirements.txt

4. Chạy ứng dụng
streamlit run app/app.py

🔗 Liên kết Dự án

Website nhóm (Drive):
https://drive.google.com/drive/folders/1WJO5ZH1D05gKEeaEvUcsH176nUwLmZSm

GitHub Repository:
https://github.com/phatdang2403-hub/Readee-Project

© 2025 Readee Project — Developed by Group 5 (VOCA AI Dept.)
