import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


df = pd.read_csv("./data/training.csv")


X = df.drop(['quality'], axis=1).as_matrix()
quality = to_categorical(df['quality'].as_matrix())

model = Sequential()
model.add(Dense(6, input_dim=4, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))

model.compile(loss='mse',
              metrics=['accuracy'],
              optimizer='rmsprop')

model.fit(X, quality, batch_size=30, epochs=10000)
