#data preprocessing

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the training set
train_dataset_path = '/Users/dawidkubicki/Documents/dataset/google_stock_price/Google_Stock_Price_Train.csv'
dataset_train = pd.read_csv(train_dataset_path)
training_set = dataset_train.iloc[:, 1:2].values

#feature scalling (recommended normalization - due to lstm)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output -> timesteps define what will be remembered of forgotten
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#building RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# RNN model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

history = regressor.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#make the predictions and visualize results

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Accuracy of a training and validation')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Loss of a training and validation')
plt.legend()

plt.show()