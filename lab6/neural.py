import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

df=pd.read_csv('ukraine-covid-cut.csv')

df.shape

data = df.filter(['new_cases'])
dataset = data.values #у масив
training_data_len = math.ceil( len(dataset) * 0.4 ) #к-сть тренувальних даних - 40%

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len , :]
x_train = []
y_train = []
step = 10
for i in range(step, len(train_data)):
  x_train.append(train_data[i - step:i, 0])
  y_train.append(train_data[i, 0])
  if (i<= step + 1):
    print(x_train)
    print(y_train)
    print()

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#модель LSTM - long short-term memory
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(1)) #кучність нейронів
model.compile(optimizer='adam', loss='mean_squared_error')

#тренування моделі
model.fit(x_train, y_train, batch_size=50, epochs=100)
steps = 10
test_data = scaled_data[training_data_len - steps: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(steps, len(test_data)):
  x_test.append(test_data[i-steps:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#прогноз
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#скв похибка
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

train = data[:training_data_len]
result = data[training_data_len:]
result['Predictions'] = predictions

#print(train)

#вивід результату
plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('days', fontsize=18)
plt.ylabel('Covid-19 cases', fontsize=18)
plt.plot(train['new_cases'], label='train data')
plt.plot(result['new_cases'], label='test data')
plt.plot(result['Predictions'], label = 'prediction', color='green')
plt.legend(loc='best')
plt.show()

model.save('model.h5')