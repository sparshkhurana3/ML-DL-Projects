import numpy as np
import pandas as pan
import pandas_datareader as pand
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = 'BTC'
labelled_against = 'INR'

start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()

data = pand.DataReader(f'{crypto_currency}-{labelled_against}', 'yahoo', start, end)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days_taken = 60
future_day_taken = 40

x_train, y_train = [], []

for x in range(prediction_days_taken, len(scaled_dataset) - future_day_taken):
    x_train.append(scaled_dataset[x - prediction_days_taken:x, 0])
    y_train.append(scaled_dataset[x + future_day_taken, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = pand.DataReader(f'{crypto_currency}-{labelled_against}', 'yahoo', test_start, test_end)
actual_price = test_data['Close'].values

total_dataset = pan.concat((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days_taken:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.fit_transform(model_input)

x_test = []

for x in range(prediction_days_taken, len(model_input)):
    x_test.append(model_input[x - prediction_days_taken:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_price, color='black', label='Actual Price')
plt.plot(predicted_prices, color='green', label='Predicted Price')
plt.title(f'{crypto_currency} Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price in INR')
plt.legend(loc='upper left')
plt.show()


real_data = [model_input[len(model_input) + 1 - prediction_days_taken:len(model_input) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print()
