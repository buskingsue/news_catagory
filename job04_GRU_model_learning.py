from tabnanny import verbose

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

# X_train, Y_train, X_test, Y_test 로드 (이전에 저장한 파일을 불러오기)
X_train = np.load('./crawling_data/news_data_X_train_max_19_wordsize 5472.npy', allow_pickle=True)
X_test = np.load('./crawling_data/news_data_X_test_max_19_wordsize_5472.npy', allow_pickle=True)
Y_train = np. load('./crawling_data/news_data_Y_train_max_19_wordsize 5472.npy', allow_pickle=True)
Y_test = np.load('./crawling_data/news_data_Y_test_max_19_wordsize 5472.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()

model.add(Embedding(5472, 300, input_length=19))
# 형태소 의미 학습 wordsize : 5293
# 차원 축소 5293 차원을 300 차원으로 줄임
# input_length == max 값 17
model.build(input_shape=(None, 19))  # 입력 데이터 크기 (None은 배치 크기)

model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
#앞뒤 관계 1d
model.add(MaxPool1D(pool_size=1))
model.add(GRU(128, return_sequences=True))  # 'activation' 제거
#장단기 기억
model.add(Dropout(0.3))
model.add(GRU(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense (128, activation='relu'))
model.add(Dense (6, activation='softmax'))
model.summary()
                    #categorical_crossentropy
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose = 0)
print('Final Test set accuracy', score[1])
model.save('./models/news_category_classfication_model_{}.h5'.format(
fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()