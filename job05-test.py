import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 데이터 로드
df = pd.read_csv('D:/2024_12_20/crawling_data/naver_headline_news_20241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['titles']
Y = df['category']

# Okt 형태소 분석기 사용
okt = Okt()

# 형태소 분석 후 X_tokenized에 저장
X_tokenized = []
for i in range(len(X)):
    X_tokenized.append(' '.join(okt.morphs(X[i], stem=True)))

print(X_tokenized[:5])

# 레이블 인코딩
with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

# 불용어 제거
stopwords = pd.read_csv('./crawling_data_2/stopwords.csv', index_col=0)
print(stopwords)

X_tokenized_no_stopwords = []
for sentence in X_tokenized:
    words = []
    for word in sentence.split():  # sentence는 이미 텍스트 형식
        if len(word) > 1 and word not in stopwords['stopword'].values:
            words.append(word)
    X_tokenized_no_stopwords.append(' '.join(words))

print(X_tokenized_no_stopwords[:5])

# Tokenizer 설정
token = Tokenizer()
token.fit_on_texts(X_tokenized_no_stopwords)
tokened_X = token.texts_to_sequences(X_tokenized_no_stopwords)

wordsize = len(token.word_index) + 1
print('wordsize:', wordsize)

print(tokened_X[:5])

# pickle로 저장된 Tokenizer 로딩 (이미 학습된 Tokenizer 사용)
with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X_tokenized_no_stopwords)
print(tokened_X[:5])

# 시퀀스 길이 조정 (최대 길이 16으로 자르기)
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 16:
        tokened_X[i] = tokened_X[i][:16]

# 패딩 처리
X_pad = pad_sequences(tokened_X, maxlen=16)
print('시퀀스 길이:', X_pad.shape)
print(X_pad[:5])

# 모델 로드
model = load_model('./models/news_category_classfication_model_0.642276406288147.h5')

# 예측
preds = model.predict(X_pad)

# 예측값을 레이블로 변환
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    predicts.append(most)

df['predict'] = predicts

# 예측 결과 출력
print(df.head(30))
