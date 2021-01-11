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
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output -> timesteps define what will be remembered of forgotten
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
#building RNN

#make the predictions and visualize results