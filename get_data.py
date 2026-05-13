from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

global dataSetSize
dataSetSize = 0

data_source = 'alphavantage' # alphavantage or kaggle

if data_source == 'alphavantage':
    # ====================== Loading Data from Alpha Vantage ==================================

    api_key = 'UU1OXTOILOW2CKF0'

    # American Airlines stock market prices
    ticker = "AAL"
    '''RELIANCE.BSE'''

    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data to this file
    file_to_save = 'stock_market_data-%s.csv'%ticker

    # If you haven't already saved data,
    # Go ahead and grab the data from the url
    # And store date, low, high, volume, close, open values to a Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
                print(df.head(10))
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)

    # If the data is already there, just load it from the CSV
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
        print(df.head(10))

def sorting(df):
    # Sort DataFrame by date
    df = df.sort_values('Date')
    print(df.head(10))
sorting(df)
'''
else:

    # ====================== Loading Data from Kaggle ==================================
    # You will be using HP's data. Feel free to experiment with other data.
    # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
    df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
    print('Loaded data from the Kaggle repository')
'''
def Visvalize(df):
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

Visvalize(df)

def split_data(df):
    # Calculate mid prices from the high and low
    high_prices = df['High'].values  # <-- use .values instead of as_matrix()
    low_prices = df['Low'].values
    mid_prices = (high_prices + low_prices) / 2.0
    total_len = mid_prices.size
    dataSetSize = total_len
    print(dataSetSize)

# 2/3 train, 1/3 test
    train_size = int(total_len * 2/3)
    test_size = total_len - train_size

    # Split data: first 2/3 for training, rest for testing
    train_data = mid_prices[:train_size]
    test_data = mid_prices[train_size:]

    print(train_data.size)
    print(test_data.size)
    print(test_size)
    return train_data, test_data

train_data,test_data=split_data(df)

def normalize(train_data,test_data):


    print("before normalizing")
    print(train_data.ndim)
    print(test_data.ndim)
    print(train_data[:5])
    print(test_data[:5])
    # Reshape to 2D
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)
   
    

    scaler = MinMaxScaler()
    smoothing_window_size = 1000

    # Chunk normalization
    for di in range(0, len(train_data), smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size, :])
        train_data[di:di+smoothing_window_size, :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

    #  After loop, normalize any remaining part (if not perfectly divisible)
    if (di + smoothing_window_size) < len(train_data):
        scaler.fit(train_data[di+smoothing_window_size:, :])
        train_data[di+smoothing_window_size:, :] = scaler.transform(train_data[di+smoothing_window_size:, :])

    # Normalize test data with the last fitted scaler
    test_data = scaler.transform(test_data)
    print("\n------------------------------------")
    print("after Normalizing")
    print(train_data.ndim)
    print(test_data.ndim)
    print(train_data[:5, :])
    print(test_data[:5, :])

    # Flatten if needed
    train_data = train_data.reshape(-1)
    test_data = test_data.reshape(-1)
    print("\n------------------------------------")
    print("after normalize reshaping")
    print(train_data.ndim)
    print(test_data.ndim)
    print(train_data[:5])
    print(test_data[:5])


   
    print("First 10 train values:", train_data[:5])
    print("First 10 test values:", test_data[:5])
    return train_data,test_data



    # Fit scaler on training data
  
train_data,test_data=normalize(train_data,test_data)
print(train_data)
print(test_data)
print(train_data.size,test_data.size)
print(train_data.max())
# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
"""
EMA = 0.0
gamma = 0.1
for ti in range(train_data.size):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA



# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)
print(all_mid_data.max())
window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
plt.figure(figsize = (8,8))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()

"""