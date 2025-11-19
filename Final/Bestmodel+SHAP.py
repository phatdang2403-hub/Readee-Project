import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import warnings
import shap

# NLP & Processing
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Model & Resampling
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. TÁI TẠO DỮ LIỆU (Data Reconstruction)
# =============================================================================
print("--- [BƯỚC 1] Tải và Chuẩn bị dữ liệu ---")

# Tải dữ liệu
try:
    df_train = pd.read_csv('dataframes/train_data.csv')
    df_val = pd.read_csv('dataframes/val_data.csv')
    df_test = pd.read_csv('dataframes/test_data.csv')

    y_train = pd.read_csv('dataframes/y_train.csv')['label']
    y_test = pd.read_csv('dataframes/y_test.csv')['label']
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file dữ liệu. Hãy kiểm tra thư mục 'dataframes'.")
    raise

# Lấy text sạch
corpus_train = df_train['text'].dropna()
corpus_test = df_test['text'].dropna()

# Mã hóa nhãn
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
class_names = list(le.classes_)
print(f"Classes: {class_names}")

# --- WORD2VEC ---
print("\n--- [BƯỚC 2] Word2Vec Vectorization ---")
tokenized_corpus = [word_tokenize(doc) for doc in corpus_train]

W2V_SIZE = 300
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=W2V_SIZE, window=5, min_count=5, workers=4)
w2v_vocab = set(w2v_model.wv.index_to_key)

def get_document_vector(doc):
    tokens = word_tokenize(doc)
    valid_tokens = [word for word in tokens if word in w2v_vocab]
    if not valid_tokens:
        return np.zeros(W2V_SIZE)
    return np.mean([w2v_model.wv[word] for word in valid_tokens], axis=0)

# Transform
print("Đang vector hóa...")
X_train_w2v = np.array([get_document_vector(doc) for doc in corpus_train])
X_test_w2v = np.array([get_document_vector(doc) for doc in corpus_test])

# --- SMOTE ---
print("\n--- [BƯỚC 3] Áp dụng SMOTE ---")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_w2v, y_train_encoded)
print(f"Phân bố lớp sau SMOTE: {Counter(y_train_res)}")

# =============================================================================
# 2. HUẤN LUYỆN MÔ HÌNH "VÔ ĐỊCH" (Final Model Training)
# =============================================================================
print("\n--- [BƯỚC 4] Huấn luyện Final Model ---")

# Cấu hình Sprint 4 (Accuracy ~80%)
final_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,     # Giữ 0.1 
    max_depth=5,           # Giữ 5 
    objective='multi:softmax',
    num_class=len(class_names),
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
final_model.fit(X_train_res, y_train_res)
end_time = time.time()
print(f"Huấn luyện xong sau {end_time - start_time:.2f} giây.")

# Lưu model
joblib.dump(final_model, 'final_model.pkl')
print("✅ Đã lưu model thành công: 'final_model.pkl'")

# =============================================================================
# 3. KIỂM TRA LẠI HIỆU SUẤT (Sanity Check)
# =============================================================================
print("\n--- [BƯỚC 5] Kiểm tra lại trên tập TEST ---")
y_pred_test = final_model.predict(X_test_w2v)

print(classification_report(y_test_encoded, y_pred_test, target_names=class_names))

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
plt.title("Final Confusion Matrix (Test Set)")
plt.show()

# =============================================================================
# 4. PHÂN TÍCH SHAP
# =============================================================================
print("\n--- [BƯỚC 6] Chạy SHAP Analysis (Fixed) ---")

# Lấy mẫu tập Test
np.random.seed(42)
indices = np.random.choice(X_test_w2v.shape[0], 200, replace=False)
X_test_sample = X_test_w2v[indices]

# 1. Tạo Explainer & Tính toán
print("Đang tính toán SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_sample)

# 2. Chuẩn hóa dữ liệu SHAP (Xử lý mảng 3 chiều & Cắt cột thừa)
processed_shap_values = []

# Kiểm tra dạng dữ liệu (List hay 3D Array)
if isinstance(shap_values, list):
    processed_shap_values = shap_values
elif len(np.array(shap_values).shape) == 3:
    # Chuyển từ (N, M, C) -> List of (N, M) cho từng lớp
    num_classes = shap_values.shape[2]
    processed_shap_values = [shap_values[:, :, i] for i in range(num_classes)]
else:
    processed_shap_values = [shap_values]

# Cắt bỏ cột bias nếu thừa
final_shap_values = []
for sv in processed_shap_values:
    if sv.shape[1] > W2V_SIZE:
        final_shap_values.append(sv[:, :W2V_SIZE])
    else:
        final_shap_values.append(sv)

# 3. Vẽ Summary Plot (Beeswarm) CHO TỪNG LỚP
print("Đang vẽ Beeswarm Plot...")
display_class_names = [f"Class {name}" for name in class_names]

for i, class_name in enumerate(display_class_names):
    # Kiểm tra an toàn
    if i >= len(final_shap_values): break
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        final_shap_values[i], 
        X_test_sample, 
        feature_names=[f"Dim_{j}" for j in range(W2V_SIZE)],
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Beeswarm: Tác động lên {class_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

# 4. Vẽ Bar Plot (Tổng quan)
print("Đang vẽ Bar Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    final_shap_values, 
    X_test_sample, 
    plot_type="bar", 
    class_names=class_names,
    feature_names=[f"Dim_{i}" for i in range(W2V_SIZE)],
    max_display=15,
    show=False
)
plt.title("SHAP Bar Plot: Tầm quan trọng trung bình của Features", fontsize=14)
plt.tight_layout()
plt.show()

print("\n✅ HOÀN TẤT TOÀN BỘ DỰ ÁN!")