import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.utils import to_categorical
from konlpy.tag import Okt
#open korea tokenizer
import jpype




df = pd.read_csv('D:/2024_12_20/crawling_data_2/naver_news_titles_20241220_1116.csv')
df.drop_duplicates(inplace=True)
print(df.head())
print('\n')
print(df.info())
print('\n')
print(df.category.value_counts())
print('\n')

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
print(okt_x)
exit()

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
