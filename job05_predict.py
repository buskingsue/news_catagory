import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.utils import to_categorical
from konlpy.tag import Okt, Kkma
#open korea tokenizer
import jpype
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


df = pd.read_csv('D:/2024_12_20/crawling_data/naver_headline_news_20241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['titles']
Y = df ['category']

print(X[0])
okt = Okt()
okt_x = okt.morphs(X[0])

print('okt_x :', okt_x)

with open('./models/encoder.pickle', 'rb') as f:
 encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

stopwords = pd.read_csv( './crawling_data_2/stopwords.csv', index_col=0)
print(stopwords)

for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence] [word]) > 1:
            if X[sentence] [word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] =' '.join(words)

print(X[:5])

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print('\nwordsize :', wordsize)
print('\n')

print(tokened_X[:5])

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 16:
        tokened_X[i] = tokened_X[i][:16]
X_pad = pad_sequences(tokened_X, 16)
print('시퀀스 길이: ',X_pad.shape)  # 패딩된 시퀀스의 길이 확인

print(X_pad[:5])

model = load_model('./models/news_category_classfication_model_0.642276406288147.h5')

preds = model.predict(X_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    predicts.append(most)
df['predict'] = predicts

print(df.head(30))


