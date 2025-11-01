# -*- coding: utf-8 -*-
# === 1. KHAI BÁO THƯ VIỆN ===
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

# Thư viện Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

# === 2. CẤU HÌNH & HẰNG SỐ ===
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

# Tên file dữ liệu
FILE_LABELED_TEXTS = "dataset/cefr_leveled_texts.csv"
FILE_WORDS_CEFR = "dataset/ENGLISH_CERF_WORDS.csv"
FILE_STORIES = "dataset/stories.csv"

# Hằng số cho data processing
STORY_QUANTILE_FILTER = 0.75  # Lọc truyện dài (giữ 75% ngắn nhất)
MIN_WORDS_IN_DICT = 5  # Số từ tối thiểu trong từ điển để gán nhãn
MIN_TEXT_LENGTH = 20  # Độ dài tối thiểu của văn bản (ký tự)

# Hằng số cho model training
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
SGD_N_EPOCHS = 50
SGD_LEARNING_RATE = 0.01

# Target labels
TARGET_LABELS = ['A1', 'A2', 'B1', 'B2']
LEVEL_SCORES = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# === 3. TẢI TÀI NGUYÊN NLTK ===
print("Đang tải tài nguyên NLTK (punkt_tab, stopwords)...")
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
print("Tải NLTK hoàn tất.")


# %%
# =============================================================================
# PHẦN 2: TẢI DỮ LIỆU
# =============================================================================

print("Đang tải 3 bộ dữ liệu...")
try:
    # 1. Tải kho ngữ liệu có nhãn
    df_cefr_texts = pd.read_csv(FILE_LABELED_TEXTS)
    print(f"Tải thành công '{FILE_LABELED_TEXTS}' ({len(df_cefr_texts)} dòng)")

    # 2. Tải từ điển CEFR
    df_words_cefr = pd.read_csv(FILE_WORDS_CEFR)
    print(f"Tải thành công '{FILE_WORDS_CEFR}' ({len(df_words_cefr)} dòng)")

    # 3. Tải kho truyện ngắn
    df_stories = pd.read_csv(FILE_STORIES, encoding='latin1')
    print(f"Tải thành công '{FILE_STORIES}' ({len(df_stories)} dòng)")

except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy tệp {e.filename}. Vui lòng đảm bảo 3 tệp ở cùng thư mục.")
except Exception as e:
    print(f"LỖI: {e}")

# %%
# ==========================================================
# 1: PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA)
# ==========================================================
print("\n--- 1: EDA ---")

# %%
# --- 1.1. EDA: Kho ngữ liệu Truyện ngắn (stories.csv) ---
print("\n[EDA 1.1] Phân tích Kho ngữ liệu (Truyện ngắn)...")

# Tính độ dài (số từ)
df_stories['word_count'] = df_stories['content'].apply(lambda x: len(str(x).split()))
print(df_stories['word_count'].describe())

# Trực quan hóa Phân bổ Độ dài truyện
plt.figure(figsize=(10, 5))
sns.histplot(df_stories['word_count'], kde=True, bins=50)
plt.title('Phân bổ Độ dài Truyện (số từ)')
plt.xlabel('Số từ')
plt.ylabel('Tần suất')
plt.show()

# Trực quan hóa Word Cloud
print("Đang tạo Word Cloud cho truyện ngắn...")
all_stories_text = " ".join(filter(None, df_stories['content']))
wordcloud = WordCloud(width=1000, height=400,
                      background_color='white',
                      stopwords=ENGLISH_STOP_WORDS).generate(all_stories_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Các từ phổ biến nhất trong truyện')
plt.show()

# %%
# --- 1.2. EDA: Dữ liệu CEFR có nhãn (cefr_leveled_texts.csv) ---
print("\n[EDA 1.2] Phân tích Dữ liệu CEFR (Văn bản có nhãn)...")

# Chuẩn hóa nhãn (ví dụ 'B1' và 'b1' là một)
df_cefr_texts['label'] = df_cefr_texts['label'].str.upper()
print("Phân bổ nhãn CEFR (văn bản có nhãn):")
print(df_cefr_texts['label'].value_counts().sort_index())

# Trực quan hóa Phân bổ nhãn (phát hiện mất cân bằng)
plt.figure(figsize=(10, 5))
sns.countplot(x='label', data=df_cefr_texts, order=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
plt.title('Phân bổ Dữ liệu theo Nhãn CEFR (Văn bản có nhãn)')
plt.xlabel('Cấp độ CEFR')
plt.ylabel('Số lượng mẫu')
plt.show()

# %%
# --- 1.3. EDA: Dữ liệu Từ vựng CEFR (ENGLISH_CERF_WORDS.csv) ---
print("\n[EDA 1.3] Phân tích Dữ liệu (Từ vựng CEFR)...")

# Chuẩn hóa nhãn
df_words_cefr['CEFR'] = df_words_cefr['CEFR'].str.upper()
print("Phân bổ nhãn CEFR (từ vựng):")
print(df_words_cefr['CEFR'].value_counts().sort_index())

# Trực quan hóa
plt.figure(figsize=(10, 5))
sns.countplot(x='CEFR', data=df_words_cefr, order=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
plt.title('Phân bổ Dữ liệu theo Nhãn CEFR (Từ vựng)')
plt.xlabel('Cấp độ CEFR')
plt.ylabel('Số lượng từ')
plt.show()

# %%
# --- 1.4. EDA: Phát hiện Chất lượng Dữ liệu ---
print("\n[EDA 1.4] Phát hiện Chất lượng Dữ liệu...")
# Dữ liệu noise đã thấy trong file cefr_leveled_texts: 'Hi!\n', '-LRB-'
noise_patterns = r'(-lrb-|-rrb-)|(\n)|(i¿)|(&nbsp;)|(<.*?>)'
df_cefr_texts['noise_found'] = df_cefr_texts['text'].str.contains(noise_patterns, na=False, case=False)
print(f"Tìm thấy {df_cefr_texts['noise_found'].sum()} mẫu văn bản chứa noise (ví dụ: \\n, -LRB-).")

print("--- KẾT THÚC EDA ---")


# %%
# =============================================================================
# 2: CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# =============================================================================
print("\n--- 2: CHUẨN BỊ DỮ LIỆU ---")

# %%
# --- 2.1. Hàm làm sạch văn bản ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Xóa tiêu đề/chân trang Gutenberg
    text = re.sub(r'\*\*\*.*?\*\*\*', ' ', text)
    
    # 2. Chuyển về chữ thường
    text = text.lower()
    
    # 3. Loại bỏ các noise
    text = re.sub(r'<.*?>', ' ', text)  # HTML tags
    text = re.sub(r'&nbsp;', ' ', text)  # &nbsp;
    text = re.sub(r'\n', ' ', text)  # Ký tự xuống dòng
    text = re.sub(r'(-lrb-|-rrb-)', ' ', text)  # -LRB-, -RRB-
    text = re.sub(r'i¿', '', text)  # Ký tự i¿
    
    # 4. Chỉ giữ chữ cái, khoảng trắng, dấu '
    text = re.sub(r'[^a-z\s\']', ' ', text)
    
    # 5. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# %%
# --- 2.2. Xây dựng từ điển tra cứu CEFR ---
print("Đang xây dựng bộ tra cứu từ vựng CEFR...")
df_words_cefr['word_clean'] = df_words_cefr['headword'].apply(clean_text)
df_words_cefr['level_clean'] = df_words_cefr['CEFR'].str.upper()
word_level_map = pd.Series(
    df_words_cefr['level_clean'].values,
    index=df_words_cefr.word_clean
).to_dict()
print(f"Đã tạo từ điển tra cứu với {len(word_level_map)} từ vựng.")

# %%
# --- 2.3. Hàm gán nhãn CEFR cho văn bản ---
def get_text_cefr_level(text, word_map):
    words = word_tokenize(text)
    score = 0
    word_count = 0

    for word in words:
        if word in word_map:
            level = word_map.get(word)
            if level in LEVEL_SCORES:
                score += LEVEL_SCORES[level]
                word_count += 1

    if word_count < MIN_WORDS_IN_DICT:
        return 'UNKNOWN'

    avg_score = score / word_count
    if avg_score < 1.8: return 'A1'
    if avg_score < 2.8: return 'A2'
    if avg_score < 3.8: return 'B1'
    if avg_score < 4.8: return 'B2'
    if avg_score < 5.8: return 'C1'
    return 'C2'

# %%
# --- 2.4. Áp dụng làm sạch và hợp nhất dữ liệu ---
print("Đang áp dụng làm sạch và hợp nhất 2 kho ngữ liệu...")

# Lọc truyện quá dài (outliers)
upper_limit = df_stories['word_count'].quantile(STORY_QUANTILE_FILTER)
print(f"Lọc truyện ngắn: Giữ lại các truyện có ít hơn {upper_limit:.0f} từ (mốc {STORY_QUANTILE_FILTER*100:.0f}%).")
df_stories_filtered = df_stories[df_stories['word_count'] <= upper_limit].copy()
print(f"Số lượng truyện ngắn còn lại sau khi lọc: {len(df_stories_filtered)}")

# 1. Xử lý văn bản có nhãn (CEFR texts)
df_cefr_texts['text_clean'] = df_cefr_texts['text'].apply(clean_text)
df_cefr_texts['label_clean'] = df_cefr_texts['label'].str.upper()
df1 = df_cefr_texts[['text_clean', 'label_clean']]

# 2. Xử lý truyện ngắn (stories)
df_stories_filtered['text_clean'] = df_stories_filtered['content'].apply(clean_text)
print("Đang gán nhãn CEFR cho truyện ngắn...")
df_stories_filtered['label_clean'] = df_stories_filtered['text_clean'].apply(
    lambda x: get_text_cefr_level(x, word_level_map)
)
df2 = df_stories_filtered[['text_clean', 'label_clean']]
print("Gán nhãn truyện ngắn hoàn tất.")

# 3. Hợp nhất hai nguồn dữ liệu
final_data = pd.concat([df1, df2], ignore_index=True)

# 4. Xử lý sau hợp nhất
final_data = final_data.dropna(subset=['text_clean', 'label_clean'])
final_data = final_data[final_data['label_clean'] != 'UNKNOWN']
final_data = final_data[final_data['text_clean'].str.len() > MIN_TEXT_LENGTH]

# Lọc chỉ lấy nhãn A1-B2 (target của dự án)
final_data = final_data[final_data['label_clean'].isin(TARGET_LABELS)]

print("\nDữ liệu sau khi làm sạch và hợp nhất:")
print(final_data.info())
print("\nPhân bổ nhãn cuối cùng (A1-B2):")
print(final_data['label_clean'].value_counts())


# %%
# --- 2.5. TF-IDF và phân chia dữ liệu (70% Train / 10% Val / 20% Test) ---
print("\nĐang thực hiện TF-IDF và phân chia Train/Validation/Test...")

X = final_data['text_clean']
y = final_data['label_clean']

# Phân chia: 70% Train, 10% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(2/3), random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Số lượng mẫu Train:      {len(y_train):4d} (70%)")
print(f"Số lượng mẫu Validation: {len(y_val):4d} (10%)")
print(f"Số lượng mẫu Test:       {len(y_test):4d} (20%)")

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Kích thước ma trận TF-IDF (Train): {X_train_tfidf.shape}")
print("--- KẾT THÚC CHUẨN BỊ DỮ LIỆU ---")


# %%
# --- 2.6. Xuất dữ liệu vào thư mục dataframes ---
print("\n--- Đang xuất dữ liệu vào thư mục dataframes ---")

import os
from scipy.sparse import save_npz

# Tạo thư mục dataframes nếu chưa tồn tại
os.makedirs('dataframes', exist_ok=True)

# 1. Xuất cleaned data (final_data)
final_data.to_csv('dataframes/cleaned_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: cleaned_data.csv")

# 2. Xuất train/val/test text và labels
# Train set
pd.DataFrame({
    'text': X_train.values,
    'label': y_train.values
}).to_csv('dataframes/train_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: train_data.csv")

# Validation set
pd.DataFrame({
    'text': X_val.values,
    'label': y_val.values
}).to_csv('dataframes/val_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: val_data.csv")

# Test set
pd.DataFrame({
    'text': X_test.values,
    'label': y_test.values
}).to_csv('dataframes/test_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: test_data.csv")

# 3. Xuất TF-IDF matrices (sparse format)
save_npz('dataframes/X_train_tfidf.npz', X_train_tfidf)
print("✅ Đã xuất: X_train_tfidf.npz")

save_npz('dataframes/X_val_tfidf.npz', X_val_tfidf)
print("✅ Đã xuất: X_val_tfidf.npz")

save_npz('dataframes/X_test_tfidf.npz', X_test_tfidf)
print("✅ Đã xuất: X_test_tfidf.npz")

# 4. Xuất labels riêng (dạng CSV)
pd.DataFrame({'label': y_train.values}).to_csv('dataframes/y_train.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: y_train.csv")

pd.DataFrame({'label': y_val.values}).to_csv('dataframes/y_val.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: y_val.csv")

pd.DataFrame({'label': y_test.values}).to_csv('dataframes/y_test.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: y_test.csv")

# 5. Lưu TF-IDF vectorizer để tái sử dụng
import pickle
with open('dataframes/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✅ Đã xuất: tfidf_vectorizer.pkl")

print("\n📦 Tất cả dữ liệu đã được xuất vào thư mục 'dataframes/'")
print(f"   - cleaned_data.csv: {len(final_data)} dòng")
print(f"   - train_data.csv: {len(y_train)} dòng")
print(f"   - val_data.csv: {len(y_val)} dòng")
print(f"   - test_data.csv: {len(y_test)} dòng")
print(f"   - TF-IDF matrices: X_train, X_val, X_test (sparse .npz format)")
print(f"   - Labels: y_train, y_val, y_test (.csv format)")
print(f"   - TF-IDF vectorizer: tfidf_vectorizer.pkl")


# %%
# =============================================================================
# 3: SO SÁNH CÁC MÔ HÌNH BASELINE
# =============================================================================
print("\n--- 3: SO SÁNH CÁC MODEL BASELINE ---")

# Lấy danh sách nhãn để dùng chung
labels_order = sorted(y.unique())


# %%
# --- Helper Function: Vẽ Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred, labels, title):
    """Vẽ confusion matrix cho model"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Nhãn Dự đoán')
    plt.ylabel('Nhãn Thực tế')
    plt.show()

# %%
# --- 3.a. Baseline model 1: Multinomial Naive Bayes ---
print("\n--- 3.a. Baseline model 1: Multinomial Naive Bayes ---")
print("Đang huấn luyện mô hình Multinomial Naive Bayes...")
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
print("Huấn luyện hoàn tất.")

# Đánh giá trên tập Validation
print("Đang đánh giá Naive Bayes trên tập Validation...")
y_pred_val_nb = model_nb.predict(X_val_tfidf)

accuracy_nb = accuracy_score(y_val, y_pred_val_nb)
print(f"\nĐộ chính xác (Accuracy) Naive Bayes: {accuracy_nb * 100:.2f}%")

print("\nBáo cáo Phân loại (Naive Bayes) - Tập Validation:")
print(classification_report(y_val, y_pred_val_nb, labels=labels_order, zero_division=0))

print("\nĐang vẽ Ma trận Nhầm lẫn (Naive Bayes)...")
plot_confusion_matrix(y_val, y_pred_val_nb, labels_order, 
                      'Confusion Matrix (Naive Bayes) - Tập Validation')



# %%
# --- 3.b. Baseline model 2: Logistic Regression ---
print("\n--- 3.b. Baseline model 2: Logistic Regression ---")
print("Đang huấn luyện mô hình Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model_lr.fit(X_train_tfidf, y_train)
print("Huấn luyện hoàn tất.")

# Đánh giá trên tập Validation
print("Đang đánh giá Logistic Regression trên tập Validation...")
y_pred_val_lr = model_lr.predict(X_val_tfidf)

accuracy_lr = accuracy_score(y_val, y_pred_val_lr)
print(f"\nĐộ chính xác (Accuracy) Logistic Regression: {accuracy_lr * 100:.2f}%")

print("\nBáo cáo Phân loại (Logistic Regression) - Tập Validation:")
print(classification_report(y_val, y_pred_val_lr, labels=labels_order, zero_division=0))

print("\nĐang vẽ Ma trận Nhầm lẫn (Logistic Regression)...")
plot_confusion_matrix(y_val, y_pred_val_lr, labels_order,
                      'Confusion Matrix (Logistic Regression) - Tập Validation')



# %%
# --- 3.c. Baseline model 3: SGDClassifier ---
print("\n--- 3.c. Baseline model 3: SGDClassifier ---")
print("Đang huấn luyện mô hình SGDClassifier (huấn luyện theo epoch)...")

# Khởi tạo SGDClassifier
model_sgd = SGDClassifier(
    loss='log_loss', 
    random_state=RANDOM_STATE, 
    eta0=SGD_LEARNING_RATE, 
    learning_rate='adaptive'
)

val_losses = []  # Lưu validation loss sau mỗi epoch
classes = labels_order

for epoch in range(SGD_N_EPOCHS):
    model_sgd.partial_fit(X_train_tfidf, y_train, classes=classes)
    
    # Tính validation loss
    y_pred_val_prob_sgd = model_sgd.predict_proba(X_val_tfidf)
    val_loss = log_loss(y_val, y_pred_val_prob_sgd, labels=classes)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{SGD_N_EPOCHS}, Validation Loss: {val_loss:.4f}")

print("Huấn luyện SGD hoàn tất.")

# Vẽ biểu đồ Loss Curve
print("\nĐang vẽ Loss Curve...")
plt.figure(figsize=(10, 6))
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve - SGDClassifier')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()

# Đánh giá SGDClassifier
print("\nĐánh giá SGDClassifier trên tập Validation...")
y_pred_val_sgd = model_sgd.predict(X_val_tfidf)

accuracy_sgd = accuracy_score(y_val, y_pred_val_sgd)
print(f"\nĐộ chính xác (Accuracy) SGDClassifier: {accuracy_sgd * 100:.2f}%")

print("\nBáo cáo Phân loại (SGDClassifier) - Tập Validation:")
print(classification_report(y_val, y_pred_val_sgd, labels=labels_order, zero_division=0))

print("\nĐang vẽ Ma trận Nhầm lẫn (SGDClassifier)...")
plot_confusion_matrix(y_val, y_pred_val_sgd, labels_order,
                      'Confusion Matrix (SGDClassifier) - Tập Validation')


# %%
# --- 3.d. Tổng kết so sánh ---
print("\n--- 3.d. TỔNG KẾT SO SÁNH BASELINE ---")
print(f"Naive Bayes Accuracy:         {accuracy_nb * 100:.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print(f"SGDClassifier Accuracy:       {accuracy_sgd * 100:.2f}%")

print("\n--- KẾT THÚC ---")