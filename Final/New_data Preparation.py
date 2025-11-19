# -*- coding: utf-8 -*-
# === 1. KHAI BÁO THƯ VIỆN ===
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import os
from scipy.sparse import save_npz
import pickle

# Thư viện Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Hằng số cho data processing
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000

# Target labels
TARGET_LABELS = ['A1', 'A2', 'B1', 'B2']
LEVEL_SCORES = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

# === 3. TẢI TÀI NGUYÊN NLTK ===
print("Đang tải tài nguyên NLTK (punkt_tab, stopwords)...")
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True) # Đảm bảo punkt được tải
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

# --- [CẢI TIẾN HÀNH ĐỘNG 1] Thêm Stopwords tùy chỉnh ---
# Loại bỏ các từ nhiễu từ Project Gutenberg và các từ chung chung không mang ý nghĩa phân loại
CUSTOM_STOP_WORDS = {
    'gutenberg', 'project', 'produced', 'end', 'start', 'chapter',
    'ebook', 'e-book', 'http', 'www', 'com', 'org', 'net',
    'said', 'says', 'one', 'would', 'could', 'should'
}
ENGLISH_STOP_WORDS.update(CUSTOM_STOP_WORDS)
print(f"Đã cập nhật Stopwords. Tổng số từ loại bỏ: {len(ENGLISH_STOP_WORDS)}")
print("Tải NLTK hoàn tất.")


# %%
# =============================================================================
# PHẦN 2: TẢI DỮ LIỆU
# =============================================================================

print("Đang tải 3 bộ dữ liệu...")
try:
    df_cefr_texts = pd.read_csv(FILE_LABELED_TEXTS)
    print(f"Tải thành công '{FILE_LABELED_TEXTS}' ({len(df_cefr_texts)} dòng)")
    df_words_cefr = pd.read_csv(FILE_WORDS_CEFR)
    print(f"Tải thành công '{FILE_WORDS_CEFR}' ({len(df_words_cefr)} dòng)")
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
print("\n--- 1: EDA (Đã rút gọn) ---")
df_stories['word_count'] = df_stories['content'].apply(lambda x: len(str(x).split()))
print("[EDA 1.1] Phân tích Kho ngữ liệu (Truyện ngắn) xong.")
df_cefr_texts['label'] = df_cefr_texts['label'].str.upper()
print("[EDA 1.2] Phân tích Dữ liệu CEFR (Văn bản có nhãn) xong.")
df_words_cefr['CEFR'] = df_words_cefr['CEFR'].str.upper()
print("[EDA 1.3] Phân tích Dữ liệu (Từ vựng CEFR) xong.")
print("--- KẾT THÚC EDA ---")


# %%
# =============================================================================
# 2: CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# =============================================================================
print("\n--- 2: CHUẨN BỊ DỮ LIỆU ---")

# --- 2.1. [CẢI TIẾN HÀNH ĐỘNG 1] Hàm làm sạch văn bản ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Xóa tiêu đề/chân trang Gutenberg
    text = re.sub(r'(\*\*\*).*?(\*\*\*)', ' ', text)
    text = re.sub(r'project gutenberg', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'end of( the)? project gutenberg', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'produced by', ' ', text, flags=re.IGNORECASE)
    
    # 2. Chuyển về chữ thường
    text = text.lower()
    
    # 3. Loại bỏ các noise
    text = re.sub(r'<.*?>', ' ', text)  # HTML tags
    text = re.sub(r'&nbsp;', ' ', text)  # &nbsp;
    text = re.sub(r'\n', ' ', text)  # Ký tự xuống dòng
    text = re.sub(r'(-lrb-|-rrb-)', ' ', text)  # -LRB-, -RRB-
    text = re.sub(r'i¿', '', text)  # Ký tự i¿
    
    # 4. Chỉ giữ chữ cái (Loại bỏ số để tránh nhiễu từ ngày tháng, số trang)
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

# --- [CẢI TIẾN HÀNH ĐỘNG 1] TF-IDF Vectorization ---
print("Đang Vector hóa (TF-IDF) với n-gram, min_df, max_df...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES, 
    stop_words=list(ENGLISH_STOP_WORDS), # Sử dụng danh sách stopwords đã tùy chỉnh
    ngram_range=(1, 2),   # Bắt cả từ đơn và cụm 2 từ
    min_df=3,             # Bỏ từ xuất hiện < 3 lần
    max_df=0.85           # Bỏ từ xuất hiện > 85% văn bản
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Kích thước ma trận TF-IDF (Train): {X_train_tfidf.shape}")
print("--- KẾT THÚC CHUẨN BỊ DỮ LIỆU ---")


# %%
# --- 2.6. Xuất dữ liệu vào thư mục dataframes ---
print("\n--- Đang xuất dữ liệu vào thư mục dataframes ---")

# Tạo thư mục dataframes nếu chưa tồn tại
os.makedirs('dataframes', exist_ok=True)

# 1. Xuất cleaned data (final_data)
final_data.to_csv('dataframes/cleaned_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: cleaned_data.csv")

# 2. Xuất train/val/test text và labels
pd.DataFrame({'text': X_train.values, 'label': y_train.values}).to_csv('dataframes/train_data.csv', index=False, encoding='utf-8')
pd.DataFrame({'text': X_val.values, 'label': y_val.values}).to_csv('dataframes/val_data.csv', index=False, encoding='utf-8')
pd.DataFrame({'text': X_test.values, 'label': y_test.values}).to_csv('dataframes/test_data.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: train/val/test data csv")

# 3. Xuất TF-IDF matrices (sparse format)
save_npz('dataframes/X_train_tfidf.npz', X_train_tfidf)
save_npz('dataframes/X_val_tfidf.npz', X_val_tfidf)
save_npz('dataframes/X_test_tfidf.npz', X_test_tfidf)
print("✅ Đã xuất: TF-IDF matrices (.npz)")

# 4. Xuất labels riêng (dạng CSV)
pd.DataFrame({'label': y_train.values}).to_csv('dataframes/y_train.csv', index=False, encoding='utf-8')
pd.DataFrame({'label': y_val.values}).to_csv('dataframes/y_val.csv', index=False, encoding='utf-8')
pd.DataFrame({'label': y_test.values}).to_csv('dataframes/y_test.csv', index=False, encoding='utf-8')
print("✅ Đã xuất: y labels csv")

# 5. Lưu TF-IDF vectorizer để tái sử dụng
with open('dataframes/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✅ Đã xuất: tfidf_vectorizer.pkl")

print("\n--- HOÀN TẤT XỬ LÝ DỮ LIỆU ---")