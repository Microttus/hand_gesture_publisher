import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique

df = pd.read_csv('Combined_data.csv')
x = df.iloc[0:4973, 0:63]
y = df.iloc[:, 63]
print(x.shape)
x = x.values.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

print(unique(y))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30)

model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(63, 1)))
model.add(Dense(32, activation="relu"))
model.add(MaxPooling1D())
model.add(LSTM(64, return_sequences=True))
model.add(Flatten())
model.add(Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
#model.summary()
model.fit(xtrain, ytrain, batch_size=16, epochs=100, verbose=0)

acc = model.evaluate(xtest, ytest)
print("Loss:", acc[0], " Accuracy:", acc[1])

pred = model.predict(xtest)
pred_y = pred.argmax(axis=-1)

cm = confusion_matrix(ytest, pred_y)
print(cm)

model.save('CNN_LSTM_Latest.h5')


