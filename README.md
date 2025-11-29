📘 Readee — AI Reading Assistant

<!-- Thay bằng link ảnh banner thực tế của nhóm nếu có -->

"Learning should adapt to you — not the other way around."

Readee là dự án nghiên cứu và phát triển (R&D) chiến lược của bộ phận AI tại VOCA, hướng tới việc xây dựng một nền tảng học tiếng Anh thông minh, cá nhân hóa hoàn toàn dựa trên dữ liệu.

Dự án giải quyết "nỗi đau" cốt lõi của người tự học: Thiếu nguồn tài liệu đọc hiểu vừa phù hợp với trình độ (CEFR), vừa hấp dẫn theo sở thích cá nhân.

🎯 Tầm nhìn & Sứ mệnh

Tầm nhìn: Xây dựng "cỗ máy tạo nội dung học tiếng Anh theo yêu cầu" (AI-driven English Content Engine), thay thế phương pháp học thụ động bằng trải nghiệm đọc chủ động.

Sứ mệnh: Ứng dụng triết lý "Comprehensible Input" (Tiếp thu dễ hiểu) để biến việc học ngôn ngữ trở nên tự nhiên, giảm áp lực và duy trì động lực học tập lâu dài.

🚀 Tính năng Cốt lõi (Key Features)

Hệ thống Readee vận hành dựa trên mô hình AI "2-trong-1":

1. 🧠 AI Phân loại (Classification Engine)

Tự động đánh giá và gán nhãn độ khó văn bản theo chuẩn CEFR (A1 - B2).

Công nghệ: XGBoost + Word2Vec.

Mục đích: Giúp người học biết chính xác trình độ của tài liệu và hỗ trợ VOCA tự động hóa quy trình phân loại kho học liệu.

2. ✍️ AI Tạo sinh (Generative Engine)

Tự động viết ra các nội dung học tập mới (truyện ngắn, hội thoại) được "đo ni đóng giày" cho từng cá nhân.

Input: Trình độ (vd: B1) + Chủ đề (vd: Công nghệ).

Output: Một câu chuyện thú vị, đúng ngữ pháp, đúng từ vựng yêu cầu.

3. 🎓 Hệ thống Bổ trợ (Interactive Learning)

Quiz Generator: Tự động tạo câu hỏi trắc nghiệm từ bài đọc.

Smart Flashcards: Trích xuất từ vựng khó trong bài để ôn tập ngay lập tức.

🛠️ Thiết lập Công nghệ (Tech Stack)

Dự án sử dụng các công nghệ tiên tiến trong Xử lý Ngôn ngữ Tự nhiên (NLP) và Học máy (ML):

Hạng mục

Công nghệ sử dụng

Mục đích

Ngôn ngữ



Ngôn ngữ lập trình chính

Xử lý Dữ liệu

Pandas, NumPy, NLTK

Tokenization, Lemmatization, Cleaning

Feature Engineering

Word2Vec (Gensim)

Nắm bắt ngữ nghĩa và ngữ cảnh từ vựng (thay vì TF-IDF)

Mô hình AI

XGBoost, Scikit-learn

Mô hình phân loại tối ưu (Final Model)

Cân bằng Dữ liệu

SMOTE (Imbalanced-learn)

Xử lý mất cân bằng dữ liệu, cải thiện nhận diện lớp khó

Giải thích AI (XAI)

SHAP

Giải thích lý do tại sao mô hình đưa ra quyết định

Ứng dụng Demo



Xây dựng Prototype tương tác nhanh chóng

📈 Kết quả Kỹ thuật Nổi bật (Proof of Concept)

Tính đến thời điểm hiện tại, dự án đã đạt được trạng thái Minimum Viable Model với những kết quả khả quan trên tập dữ liệu kiểm thử độc lập:

Dữ liệu: Xây dựng thành công bộ dữ liệu chuẩn hóa gồm 1,235 mẫu văn bản đã được làm sạch và cân bằng (A1-B2).

Hiệu suất Mô hình (XGBoost + Word2Vec + SMOTE):

Độ chính xác tổng thể (Accuracy): ~76% (Chấp nhận đánh đổi để tăng độ phủ cho các lớp khó).

Recall lớp khó (B1): Đạt 71% (Cải thiện vượt bậc so với mức 37% của các mô hình cũ, giải quyết triệt để vấn đề bỏ sót bài học phù hợp).

Tốc độ phản hồi: < 1 giây/request.

📂 Cấu trúc Dự án

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


🏃 Hướng dẫn Cài đặt & Chạy Demo

Để trải nghiệm Readee trên máy cục bộ, hãy làm theo các bước sau:

Bước 1: Clone Repository

git clone [https://github.com/phatdang2403-hub/Readee-Project.git](https://github.com/phatdang2403-hub/Readee-Project.git)
cd Readee-Project


Bước 2: Thiết lập môi trường

Khuyến nghị sử dụng môi trường ảo để tránh xung đột thư viện.

# Tạo môi trường ảo (Windows)
python -m venv venv
.\venv\Scripts\activate

# Hoặc trên Mac/Linux
source venv/bin/activate


Bước 3: Cài đặt thư viện

pip install -r requirements.txt


Bước 4: Chạy ứng dụng

streamlit run app/app.py


Sau khi chạy, truy cập địa chỉ http://localhost:8501 trên trình duyệt để sử dụng.

👥 Đội ngũ Phát triển (Readee Team - Group 5)

Dự án được thực hiện bởi đội ngũ chuyên trách thuộc bộ phận AI của VOCA:

Thành viên

Vai trò & Trách nhiệm

Kha

Project Leader - Định hướng chiến lược, Quản lý dự án.

Thịnh

Data Engineer - Xử lý dữ liệu, ETL Pipeline.

Đạt

AI Developer - Phát triển mô hình, Tối ưu hóa thuật toán.

Phát

Technical Analyst - Phân tích kỹ thuật, AI Canvas, XAI.

T.Minh

Resource Manager - Quản lý nguồn lực, Tài liệu dự án.

K.Linh

UI/UX & Comms - Thiết kế giao diện, Truyền thông.

Hằng

Presentation - Điều phối thuyết trình & Tiến độ.

🔗 Liên kết Dự án

Website Nhóm (Tài liệu): Google Drive Link

Mã nguồn (GitHub): phatdang2403-hub/Readee-Project

© 2025 Readee Project - Developed by Group 5 (VOCA AI Dept).
