import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
from keras.models import load_model 
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Prediction App')

user_input = st.text_input("Enter Stock Ticker", 'AAPL')
import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'


df = yf.download(user_input, start=start, end=end)


st.line_chart(df.Close) 
st.line_chart(df.Volume)
st.subheader("Date from 2010-01-01 to 2019-12-31")
st.write(df.describe())

#visualization 

st.subheader('Closing Price vs Time Chart')

fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
st.subheader('Volume vs Time Chart')

fig = plt.figure(figsize=(12,6))
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100, color='red')    
plt.plot(ma200, color='green')
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

#load model

model = load_model('saved_model.h5')

#testing data set

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = model.predict(x_test)
y_test = y_test*scale_factor


#final graph

st.subheader('Predictions vs Real Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, color='red', label='Real Price')
plt.plot(y_pred, color='blue', label='Predicted Price')
plt.title('Predictions vs Real Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
