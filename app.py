import streamlit as st
from urllib.parse import urlparse
import joblib
import numpy as np
import math
import pandas as pd
from collections import Counter


# Hàm trích xuất domain từ URL
def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc.rsplit('.', 1)[0] if len(parsed_url.netloc.rsplit('.', 1)) > 1 else parsed_url.netloc

# Các hàm đặc trưng (đảm bảo rằng các hàm này đã được định nghĩa đầy đủ)
#Hàm trả vể entropy của tên miền
def calculate_entropy(domain):
    prob = [domain.count(c) / len(domain) for c in set(domain)]
    entropy = -sum(p * math.log2(p) for p in prob if p > 0)
    return entropy

df_probabilities = pd.read_csv('character_probabilities.csv')
character_probabilities = dict(zip(df_probabilities['Character'], df_probabilities['Probability']))

#Hàm tính giá trị kì vọng
def calculate_expected_value(domain, character_probabilities):
    # Bước 1: Tính tần suất xuất hiện của các ký tự trong tên miền
    character_counts = Counter(domain)
    
    # Bước 2: Tính tử số của công thức
    numerator = sum(character_counts[char] * character_probabilities.get(char, 0) 
                    for char in character_counts)
    
    # Bước 3: Tính mẫu số của công thức
    denominator = sum(character_counts.values())  # Tổng số ký tự trong domain
    
    # Bước 4: Tính giá trị kỳ vọng
    expected_value = numerator / denominator if denominator != 0 else 0
    
    return expected_value

#Hàm trả về độ dài của tên miền
def domain_length(domain):
    return len(domain)

#Tỉ lệ nguyên âm - phụ âm
def vowel_consonant_ratio(domain):
    vowels = set('aeiou')
    num_vowels = sum(1 for char in domain if char in vowels)
    num_consonants = sum(1 for char in domain if char.isalpha() and char not in vowels)
    if num_consonants == 0:  # Tránh chia cho 0
        return 0
    return num_vowels / num_consonants

#Tỉ lệ kí tự số và chữ
def digit_letter_ratio(domain):
    num_digits = sum(1 for char in domain if char.isdigit())
    num_letters = sum(1 for char in domain if char.isalpha())
    if num_letters == 0:  # Tránh chia cho 0
        return 0
    return num_digits / num_letters

#Tỷ lệ kí tự đặc biệt
def special_char_ratio(domain):
    special_chars = set('!@#$%^&*()_+-=[]{}|;:",.<>?/')
    num_special_chars = sum(1 for char in domain if char in special_chars)
    return num_special_chars/domain_length(domain)

# Đường dẫn tới tệp tri-grams và bi-grams mặc định
default_trigram_file_path = "DS_tri_gram.csv"
default_bigram_file_path = "DS_bi_gram.csv"

# Đọc tệp tri-grams vào DataFrame
df_trigrams = pd.read_csv(default_trigram_file_path)
top_100_trigrams_set = set(df_trigrams['tri_gram'])

# Đọc tệp bi-grams vào DataFrame
df_bigrams = pd.read_csv(default_bigram_file_path)
top_100_bigrams_set = set(df_bigrams['bi_gram'])

# Hàm tính n-grams
def calculate_ngrams(domain, n):
    # Đảm bảo domain là chuỗi
    domain = str(domain)
    n_grams = [domain[i:i+n] for i in range(len(domain)-n+1)]
    return n_grams

# Hàm đếm tri-grams
def count_trigram(domain):
    tri_grams = calculate_ngrams(domain, 3)  # Sử dụng n=3 cho tri-grams
    existing_trigrams = [trigram for trigram in tri_grams if trigram in top_100_trigrams_set]
    count_existing = len(existing_trigrams)
    return count_existing

# Hàm đếm bi-grams
def count_bigram(domain):
    bi_grams = calculate_ngrams(domain, 2)  # Sử dụng n=2 cho bi-grams
    existing_bigrams = [bigram for bigram in bi_grams if bigram in top_100_bigrams_set]
    count_existing = len(existing_bigrams)
    return count_existing


#Tính trung bình số lượng bi_gram phổ biến trong tên miền
def avg_bigram(domain):
    amount_bigram = len(calculate_ngrams(domain,2))
    common_bigram = count_bigram(domain)
    if amount_bigram == 0:
        return 0  # Tránh phép chia cho 0
    return common_bigram/amount_bigram

#Tính trung bình số lượng tri_gram phổ biến trong tên miền
def avg_trigram(domain):
    amount_trigram = len(calculate_ngrams(domain,3))
    common_trigram = count_trigram(domain)
    if amount_trigram == 0:
        return 0  # Tránh phép chia cho 0
    return common_trigram/amount_trigram

# Tải mô hình đã lưu
model = joblib.load('rf_model.pkl')


# Tiêu đề của ứng dụng
st.title("Dự đoán domain với mô hình Random Forest")

# Ô nhập URL từ người dùng
url_input = st.text_input("Nhập URL:", value="")

# Khi người dùng nhấn nút "Dự đoán"
if st.button("Dự đoán"):
    if url_input:
        # Trích xuất domain từ URL
        domain = extract_domain(url_input)
        st.write(f"Domain trích xuất từ URL: {domain}")
        
        # Giả sử bạn có các xác suất ký tự cho hàm expected value
        # character_probabilities = {}

        # Tạo các đặc trưng từ domain
        new_data_feature = [[round(calculate_entropy(domain), 4), 
                             calculate_expected_value(domain, character_probabilities),
                             domain_length(domain), 
                             vowel_consonant_ratio(domain), 
                             digit_letter_ratio(domain), 
                             special_char_ratio(domain), 
                             count_bigram(domain), 
                             count_trigram(domain), 
                             avg_bigram(domain), 
                             avg_trigram(domain)]]

        # Dự đoán cho domain mới
        # Dự đoán kết quả cho domain mới
        prediction = model.predict(new_data_feature)

        # Kiểm tra kết quả dự đoán và in ra Đúng hoặc Sai
        if prediction[0] == 0:
            st.write('Kết quả dự đoán: không phải DGA botnet')
        else:
            st.write('Kết quả dự đoán: là DGA botnet')





