import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
import joblib

# NLP & Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# XGBoost & SMOTE
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# SHAP
import shap

# Cấu hình
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# =============================================================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# =============================================================================
print("--- [BƯỚC 1] Tải và Chuẩn bị dữ liệu ---")

# Tải dữ liệu thô
df_train = pd.read_csv('dataframes/train_data.csv')
df_val = pd.read_csv('dataframes/val_data.csv')
df_test = pd.read_csv('dataframes/test_data.csv')

# Tải nhãn
y_train = pd.read_csv('dataframes/y_train.csv')['label']
y_val = pd.read_csv('dataframes/y_val.csv')['label']
y_test = pd.read_csv('dataframes/y_test.csv')['label']

# Lấy text và loại bỏ null
corpus_train = df_train['text'].dropna()
corpus_val = df_val['text'].dropna()
corpus_test = df_test['text'].dropna()

# Mã hóa nhãn
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)
class_names = list(le.classes_)
print(f"Classes: {class_names}")

# =============================================================================
# 2. VECTOR HÓA (WORD2VEC)
# =============================================================================
print("\n--- [BƯỚC 2] Word2Vec Vectorization ---")

# Tokenize
tokenized_corpus = [word_tokenize(doc) for doc in corpus_train]

# Huấn luyện Word2Vec
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
X_train_w2v = np.array([get_document_vector(doc) for doc in corpus_train])
X_val_w2v = np.array([get_document_vector(doc) for doc in corpus_val])
X_test_w2v = np.array([get_document_vector(doc) for doc in corpus_test])
print(f"Kích thước X_train: {X_train_w2v.shape}")

# =============================================================================
# 3. CÂN BẰNG DỮ LIỆU (SMOTE)
# =============================================================================
print("\n--- [BƯỚC 3] Áp dụng SMOTE ---")
print(f"Trước SMOTE: {Counter(y_train_encoded)}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_w2v, y_train_encoded)

print(f"Sau SMOTE:   {Counter(y_train_res)}")

# =============================================================================
# 4. TUNING (RANDOMIZED SEARCH CV)
# =============================================================================
print("\n--- [BƯỚC 4] Tinh chỉnh Siêu tham số (Hyperparameter Tuning) ---")

# Định nghĩa không gian tham số
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb = XGBClassifier(objective='multi:softmax', num_class=len(class_names), n_jobs=-1, random_state=42)

# Sử dụng F1_MACRO để tối ưu cho lớp thiểu số (B1)
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,             # Số lượng tổ hợp thử nghiệm
    scoring='f1_macro',    # Không dùng accuracy
    cv=3,                  # Cross-validation 3-fold
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Đang tìm kiếm bộ tham số tốt nhất...")
start_time = time.time()
random_search.fit(X_train_res, y_train_res)
end_time = time.time()

print(f"Tuning hoàn tất sau {end_time - start_time:.2f} giây.")
print(f"Best Params: {random_search.best_params_}")
print(f"Best CV F1-Macro Score: {random_search.best_score_:.4f}")

# =============================================================================
# 5. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT (FINAL EVALUATION)
# =============================================================================
print("\n--- [BƯỚC 5] Đánh giá trên tập TEST ---")

best_model = random_search.best_estimator_

# Dự đoán trên tập Test
y_pred_test = best_model.predict(X_test_w2v)

print("\nBáo cáo phân loại (Test Set):")
print(classification_report(y_test_encoded, y_pred_test, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
plt.title("Confusion Matrix - Tuned XGBoost (Test Set)")
plt.show()

# Lưu mô hình tốt nhất
joblib.dump(best_model, 'best_xgboost(tuned).pkl')
print("Đã lưu mô hình vào 'best_xgboost(tuned).pkl'")

# =============================================================================
# 6. GIẢI THÍCH MÔ HÌNH (SHAP ANALYSIS)
# =============================================================================
print("\n--- [BƯỚC 6] Phân tích SHAP (Explainability) ---")

# Lấy mẫu dữ liệu Test để chạy SHAP
# Lấy 100 mẫu ngẫu nhiên
indices = np.random.choice(X_test_w2v.shape[0], 100, replace=False)
X_test_sample = X_test_w2v[indices]

# Khởi tạo Explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sample)

# 6.1. Summary Plot (Tổng quan tác động Feature)
print("Đang vẽ Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_test_sample, class_names=class_names, feature_names=[f"Dim_{i}" for i in range(W2V_SIZE)])
plt.show()

# 6.2. Bar Plot (Tầm quan trọng của Feature)
print("Đang vẽ Bar Plot...")
plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", class_names=class_names, feature_names=[f"Dim_{i}" for i in range(W2V_SIZE)])
plt.show()

print("\n✅ HOÀN TẤT TUNING")