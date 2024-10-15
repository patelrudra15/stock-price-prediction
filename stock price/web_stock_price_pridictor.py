import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('Stock Prediction Model.keras')

st.header('Stock Price Pridictor')

stock = st.text_input('Enter Stock Symbole','GOOG')
start = '2000-01-01'
end = '2024-10-10'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days =  data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader(' Price vs Moving Average 50 Days')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs Moving Average 50 Days vs Moving Average 100 Days')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs Moving Average 100 Days vs Moving Average 200 Days')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)

st.subheader('Price vs Moving Average 200 Days vs Moving Average 250 Days')
ma_250_days = data.Close.rolling(250).mean()
fig4 = plt.figure(figsize=(12,6))
plt.plot(ma_200_days,'r')
plt.plot(ma_250_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig4)


x = []
y = []

for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

scale = predict * scale
y = y*scale

st.subheader('OriginalPrice vs Pridicted Price')
fig5 = plt.figure(figsize=(12,6))
plt.plot(predict,'r',label='Original Price')
plt.plot(y,'g', label ='Pridicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig5)