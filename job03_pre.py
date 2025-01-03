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



df = pd.read_csv('D:/2024_12_20/crawling_data_2/naver_news_titles_20241220_1401.csv')
df.drop_duplicates(inplace=True)
print(df.head())
print('\n')
print(df.info())
print('\n')
print(df.category.value_counts())
print('\n')
df = df.reset_index(drop=True)  # 인덱스를 재설정

X = df['title']
Y = df['category']

print(X[0])
okt = Okt()
okt_x = okt.morphs(X[0])
'''
이 코드는 okt 객체의 morphs 메서드를 사용하여 X[0]의 형태소 분석 결과를 okt_x에 저장하는 코드입니다.

okt_x = : okt_x라는 변수에 값을 할당하고 있습니다.
okt.morphs(X[0]) : okt 객체의 morphs 메서드는 주어진 텍스트 X[0]을 형태소 단위로 분해합니다. X[0]은 입력 데이터 리스트에서 첫 번째 요소입니다.
즉, 이 코드는 X[0]에 있는 텍스트를 형태소 단위로 분석하여 그 결과를 okt_x에 저장하는 역할을 합니다.
'''
print('okt_x :', okt_x)

# kkma = Kkma()
# kkma_x = kkma.morphs(X[0])
# print('kkma :', kkma_x)
# exit()

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])
label = encoder.classes_
print(label)
with open('./models/encoder.pickle', 'wb') as f:
    #쓰기모드 바이너리
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)
#불용어 : 학습이 도움이 안되는 쓸모없는 문자 대명사 감탄사..
stopwords = pd.read_csv('./crawling_data_2/stopwords.csv', index_col=0)
print(stopwords)


for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
           if X[sentence][word] not in list(stopwords['stopword']):
               words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print('\nwordsize :', wordsize)
print('\n')

print(tokened_X[:5])
print('이 코드는 tokened_X라는 변수의 처음 5개의 요소를 출력하는 역할을 합니다.\n')


print('\n')
#최대값 찾기 알고리즘
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print('max값: ', max)
print('\n')

X_pad = pad_sequences(tokened_X, max)
print(X_pad)
print(len(X_pad[0]))

X_train, X_test, Y_train, Y_test = train_test_split( X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

np.save('./crawling_data/news_data_X_train_max_{}_wordsize {}'.format( max, wordsize), X_train)
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize {}'.format( max, wordsize), Y_train)
np.save('./crawling_data/news_data_X_test_max_{}_wordsize_{}'.format( max, wordsize), X_test)
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize {}'.format( max, wordsize), Y_test)