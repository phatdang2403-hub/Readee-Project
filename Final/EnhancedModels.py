import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from scipy.sparse import load_npz

# Scikit-learn & Utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost & LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

# Cấu hình hiển thị
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. TẢI DỮ LIỆU (TỪ SPRINT 2)
# =============================================================================
print("--- [BƯỚC 1] Tải dữ liệu thô từ Sprint 2 ---")

# Tải văn bản thô (để huấn luyện Word2Vec)
df_train = pd.read_csv('dataframes/train_data.csv')
df_val = pd.read_csv('dataframes/val_data.csv')
df_test = pd.read_csv('dataframes/test_data.csv')

# Tải nhãn
y_train = pd.read_csv('dataframes/y_train.csv')['label']
y_val = pd.read_csv('dataframes/y_val.csv')['label']
y_test = pd.read_csv('dataframes/y_test.csv')['label']

# Lấy cột text và loại bỏ giá trị null
corpus_train = df_train['text'].dropna()
corpus_val = df_val['text'].dropna()
corpus_test = df_test['text'].dropna()

print(f"Số lượng mẫu Train: {len(corpus_train)}")
print(f"Số lượng mẫu Val:   {len(corpus_val)}")
print(f"Số lượng mẫu Test:  {len(corpus_test)}")

# =============================================================================
# 2. MÃ HÓA NHÃN (LABEL ENCODING)
# =============================================================================
print("\n--- [BƯỚC 2] Mã hóa nhãn ---")
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)
class_names = le.classes_
print(f"Nhãn đã mã hóa: {class_names}")

# =============================================================================
# 3. VECTOR HÓA BẰNG WORD2VEC
# =============================================================================
print("\n--- [BƯỚC 3] Tạo features bằng Word2Vec ---")

# Tokenize
print("Đang token hóa dữ liệu huấn luyện...")
tokenized_corpus = [word_tokenize(doc) for doc in corpus_train]

# Huấn luyện Word2Vec Model
W2V_SIZE = 300
print(f"Đang huấn luyện Word2Vec model (size={W2V_SIZE})...")
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=W2V_SIZE, window=5, min_count=5, workers=4)
w2v_vocab = set(w2v_model.wv.index_to_key)

# Hàm chuyển đổi văn bản thành vector trung bình
def get_document_vector(doc, model, num_features):
    tokens = word_tokenize(doc)
    valid_tokens = [word for word in tokens if word in w2v_vocab]
    if not valid_tokens:
        return np.zeros(num_features)
    return np.mean([model.wv[word] for word in valid_tokens], axis=0)

# Áp dụng vector hóa
print("Đang biến đổi Train/Val/Test thành vector...")
X_train_w2v = np.array([get_document_vector(doc, w2v_model, W2V_SIZE) for doc in corpus_train])
X_val_w2v = np.array([get_document_vector(doc, w2v_model, W2V_SIZE) for doc in corpus_val])
X_test_w2v = np.array([get_document_vector(doc, w2v_model, W2V_SIZE) for doc in corpus_test])
print("Vector hóa hoàn tất.")

# =============================================================================
# 4. ÁP DỤNG SMOTE (CÂN BẰNG DỮ LIỆU)
# =============================================================================
print("\n--- [BƯỚC 4] Áp dụng SMOTE ---")
print(f"Phân bố lớp gốc (Train): {Counter(y_train_encoded)}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_w2v, y_train_encoded)

print(f"Phân bố lớp sau SMOTE:   {Counter(y_train_resampled)}")
print(f"Kích thước X_train mới: {X_train_resampled.shape}")

# =============================================================================
# 5. HUẤN LUYỆN VÀ SO SÁNH 5 MÔ HÌNH
# =============================================================================
print("\n--- [BƯỚC 5] Chạy thử nghiệm 5 Mô hình ---")

# Định nghĩa danh sách mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    
    "LinearSVC": LinearSVC(random_state=42, dual='auto'),
    
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                             objective='multi:softmax', num_class=len(class_names), 
                             n_jobs=-1, random_state=42),
    
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, num_class=len(class_names),
                               objective='multiclass', n_jobs=-1, random_state=42, verbose=-1)
}

# Vòng lặp huấn luyện và đánh giá
for name, model in models.items():
    print(f"\n{'='*20} {name} {'='*20}")
    
    # 1. Huấn luyện (Trên tập đã SMOTE)
    print(f"Đang huấn luyện {name}...")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    end_time = time.time()
    print(f"Thời gian huấn luyện: {end_time - start_time:.2f} giây")
    
    # 2. Dự đoán (Trên tập Validation GỐC)
    y_pred = model.predict(X_val_w2v)
    
    # 3. Báo cáo
    print(f"Kết quả trên tập Validation ({name}):")
    print(classification_report(y_val_encoded, y_pred, target_names=class_names))
    
    # 4. Vẽ Confusion Matrix
    cm = confusion_matrix(y_val_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# =============================================================================
# 6. [PHẦN BỔ SUNG] CHẠY SONG SONG VỚI TF-IDF + SMOTE
# =============================================================================
print("\n" + "="*50)
print("   BƯỚC 6: THỬ NGHIỆM MỞ RỘNG (TF-IDF + SMOTE)")
print("="*50)

# 6.1. Tải dữ liệu TF-IDF
print("Đang tải dữ liệu TF-IDF từ file .npz...")
try:
    X_train_tfidf = load_npz('dataframes/X_train_tfidf.npz')
    X_val_tfidf = load_npz('dataframes/X_val_tfidf.npz')
    print(f"Kích thước X_train_tfidf gốc: {X_train_tfidf.shape}")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file .npz. Hãy chắc chắn bạn đã chạy code Sprint 2.")
    exit()

# 6.2. Áp dụng SMOTE cho TF-IDF
print("\nĐang áp dụng SMOTE lên dữ liệu TF-IDF (Quá trình này có thể mất 1-2 phút)...")
smote_tfidf = SMOTE(random_state=42)
X_train_tfidf_res, y_train_tfidf_res = smote_tfidf.fit_resample(X_train_tfidf, y_train_encoded)

print(f"Phân bố lớp sau SMOTE (TF-IDF): {Counter(y_train_tfidf_res)}")
print(f"Kích thước X_train_tfidf mới: {X_train_tfidf_res.shape}")

# 6.3. Huấn luyện và Đánh giá lại 5 mô hình trên TF-IDF
print("\n--- So sánh hiệu suất trên đặc trưng TF-IDF ---")

# Khởi tạo lại models để đảm bảo không bị lẫn lộn với model Word2Vec cũ
models_tfidf = {
    "Logistic Regression (TF-IDF)": LogisticRegression(max_iter=1000, random_state=42),
    
    "LinearSVC (TF-IDF)": LinearSVC(random_state=42, dual='auto'),
    
    "Random Forest (TF-IDF)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    
    "XGBoost (TF-IDF)": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, 
                                      objective='multi:softmax', num_class=len(class_names), 
                                      n_jobs=-1, random_state=42),
    
    "LightGBM (TF-IDF)": LGBMClassifier(n_estimators=200, learning_rate=0.1, num_class=len(class_names),
                                        objective='multiclass', n_jobs=-1, random_state=42, verbose=-1)
}

for name, model in models_tfidf.items():
    print(f"\n>>> Đang huấn luyện: {name}")
    start_time = time.time()
    
    # Huấn luyện trên dữ liệu TF-IDF đã SMOTE
    model.fit(X_train_tfidf_res, y_train_tfidf_res)
    end_time = time.time()
    
    # Dự đoán trên tập Validation TF-IDF gốc
    y_pred = model.predict(X_val_tfidf)
    
    print(f"Thời gian huấn luyện: {end_time - start_time:.2f}s")
    print(classification_report(y_val_encoded, y_pred, target_names=class_names))
    
    # Vẽ Confusion Matrix (Dùng màu Xanh Lá để phân biệt)
    cm = confusion_matrix(y_val_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Greens, ax=ax, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

print("\n✅✅ HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH ✅✅")