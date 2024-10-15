import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from  keras.models  import  load_model

st.title('Stock Price Pridictor App')

stock = st.text_input("Enter The Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

google_data = yf.download(stock, start,end)

model = load_model("latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'orange')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_data)
    return fig  

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 Days and MA for 250 days')
st.pyplot(plot_graph((15,6),google_data['MA_for_250_days'],google_data,1,google_data['MA_for_250_days'] ))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []
 
for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    
    
x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'original_test_data' : inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    # The index is now generated using the length of inv_y_test
    index = range(len(inv_y_test))
)
st.subheader("Original values vs Pridicted values")
st.write(ploting_data) 

st.subheader('Original Close Price vs Pridicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used","Original Test data","predicted Test data"])
st.pyplot(fig)