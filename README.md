# 📘 Readee — AI Reading Assistant

![Readee Banner](https://via.placeholder.com/1200x350?text=Readee+AI+Project+Banner) > **"Learning should adapt to you — not the other way around."**

**Readee** là dự án nghiên cứu và phát triển (R&D) chiến lược của bộ phận AI tại **VOCA**, hướng tới việc xây dựng một nền tảng học tiếng Anh thông minh, cá nhân hóa hoàn toàn dựa trên dữ liệu.

Dự án giải quyết "nỗi đau" cốt lõi của người tự học: **Thiếu nguồn tài liệu đọc hiểu vừa phù hợp với trình độ (CEFR), vừa hấp dẫn theo sở thích cá nhân.**

---

## 🎯 Tầm nhìn & Sứ mệnh

* **Tầm nhìn:** Xây dựng "cỗ máy tạo nội dung học tiếng Anh theo yêu cầu" (AI-driven English Content Engine), thay thế phương pháp học thụ động bằng trải nghiệm đọc chủ động.
* **Sứ mệnh:** Ứng dụng triết lý **"Comprehensible Input"** (Tiếp thu dễ hiểu) để biến việc học ngôn ngữ trở nên tự nhiên, giảm áp lực và duy trì động lực học tập lâu dài.

---

## 🚀 Tính năng Cốt lõi (Key Features)

Hệ thống Readee vận hành dựa trên mô hình AI "2-trong-1":

### 1. 🧠 AI Phân loại (Classification Engine)
Tự động đánh giá và gán nhãn độ khó văn bản theo chuẩn **CEFR (A1 - B2)**.
* Giúp người học biết chính xác trình độ của tài liệu.
* Hỗ trợ VOCA tự động hóa quy trình phân loại kho học liệu khổng lồ.

### 2. ✍️ AI Tạo sinh (Generative Engine)
Tự động viết ra các nội dung học tập mới (truyện ngắn, hội thoại) được "đo ni đóng giày" cho từng cá nhân.
* **Input:** Trình độ (vd: B1) + Chủ đề (vd: Công nghệ).
* **Output:** Một câu chuyện thú vị, đúng ngữ pháp, đúng từ vựng yêu cầu.

### 3. 🎓 Hệ thống Bổ trợ (Interactive Learning)
* **Quiz Generator:** Tự động tạo câu hỏi trắc nghiệm từ bài đọc.
* **Smart Flashcards:** Trích xuất từ vựng khó trong bài để ôn tập ngay lập tức.

---

## 🛠️ Thiết lập Công nghệ (Tech Stack)

Dự án sử dụng các công nghệ tiên tiến trong Xử lý Ngôn ngữ Tự nhiên (NLP) và Học máy (ML):

| Hạng mục | Công nghệ sử dụng |
| :--- | :--- |
| **Ngôn ngữ** | ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) |
| **Xử lý Dữ liệu** | Pandas, NumPy, NLTK (Tokenization, Lemmatization) |
| **Feature Engineering** | **Word2Vec** (Gensim), TF-IDF (Scikit-learn) |
| **Mô hình AI** | **XGBoost**, Logistic Regression, Random Forest |
| **Cân bằng Dữ liệu** | **SMOTE** (Imbalanced-learn) để xử lý mất cân bằng lớp |
| **Giải thích AI (XAI)** | **SHAP** (Để giải thích quyết định của mô hình) |
| **Ứng dụng Demo** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit) |
| **Quản lý Code** | GitHub |

---

## 📈 Kết quả Kỹ thuật Nổi bật (Sprint 4 Status)

Dự án đã hoàn thành giai đoạn **Proof of Concept (PoC)** với những kết quả khả quan trên tập dữ liệu kiểm thử (Test Set):

* **Dữ liệu:** Xây dựng thành công bộ dữ liệu chuẩn hóa gồm **1,235 mẫu văn bản** đã được làm sạch và cân bằng (A1-B2).
* **Hiệu suất Mô hình (XGBoost + Word2Vec + SMOTE):**
    * **Độ chính xác (Accuracy):** ~76% (Chấp nhận đánh đổi để tăng độ phủ).
    * **Recall lớp khó (B1):** Đạt **71%** (Cải thiện vượt bậc so với mức 37% của các mô hình cũ, giải quyết triệt để vấn đề bỏ sót bài học phù hợp).
    * **Tốc độ phản hồi:** < 1 giây/request.

---

## 📂 Cấu trúc Dự án

```bash
Readee-Project/
├── app/
│   ├── app.py                 # Mã nguồn ứng dụng Demo Streamlit
│   ├── models/                # Chứa file model (.pkl) và vectorizer
│   └── utils/                 # Các hàm xử lý phụ trợ
├── notebooks/
│   ├── 1_EDA.ipynb            # Phân tích khám phá dữ liệu
│   ├── 2_Preprocessing.ipynb  # Làm sạch và chuẩn hóa dữ liệu
│   ├── 3_Baseline_Model.ipynb # Huấn luyện Logistic Regression
│   └── 4_Final_Model.ipynb    # Tối ưu hóa XGBoost + SMOTE + Word2Vec
├── data/
│   ├── raw/                   # Dữ liệu thô
│   └── processed/             # Dữ liệu đã làm sạch (Cleaned & Balanced)
├── docs/                      # Tài liệu báo cáo, AI Canvas, Slide
├── requirements.txt           # Danh sách thư viện
└── README.md                  # File giới thiệu này
