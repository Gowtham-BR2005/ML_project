from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# Configuration
data_source = 'alphavantage'  # or 'csv' if you have existing data

# ====================== TOGGLE PREDICTION DAYS HERE ==================================
# Change this value to: 1, 3, 7, 30, or 90 days
PREDICTION_DAYS = 90  # Options: 1, 3, 7, 30, 90

LOOK_BACK = 60  # Use 60 days of history to predict

# ====================== Loading Data ==================================
if data_source == 'alphavantage':
    api_key = 'UU1OXTOILOW2CKF0'
    ticker = "AAL"
    
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = 'stock_market_data-%s.csv'%ticker
    
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
else:
    # Load from your existing CSV
    df = pd.read_csv('stock_market_data-AAL.csv')

# Sort by date
df = df.sort_values('Date')
df = df.reset_index(drop=True)
print(f"Total data points: {len(df)}")
print(df.head())

# ====================== Visualization ==================================
def visualize_data(df):
    plt.figure(figsize=(18, 6))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].iloc[::500], rotation=45)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Mid Price', fontsize=14)
    plt.title('Stock Price History')
    plt.tight_layout()
    plt.show()

visualize_data(df)

# ====================== Data Preparation ==================================
# Calculate mid prices
high_prices = df['High'].values
low_prices = df['Low'].values
mid_prices = (high_prices + low_prices) / 2.0
mid_prices = mid_prices.reshape(-1, 1)

print(f"Dataset size: {len(mid_prices)}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(mid_prices)

# Split into train and test (80/20 split)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# ====================== Create Sequences ==================================
def create_sequences(data, look_back=60):
    """
    Create sequences for LSTM training
    X: input sequences of 'look_back' days
    y: target value (next day's price)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, LOOK_BACK)
X_test, y_test = create_sequences(test_data, LOOK_BACK)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ====================== Build LSTM Model ==================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = build_lstm_model((X_train.shape[1], 1))
model.summary()

# ====================== Train Model ==================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====================== Evaluate Model ==================================
# Make predictions on test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = np.mean((test_predictions - y_test_actual)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_predictions - y_test_actual))

print(f"\nTest Set Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model MAE')
plt.tight_layout()
plt.show()

# ====================== Predict Future 90 Days ==================================
def predict_future(model, last_sequence, scaler, days=90):
    """
    Predict future stock prices
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Reshape for prediction
        current_input = current_sequence.reshape(1, LOOK_BACK, 1)
        
        # Predict next day
        next_pred = model.predict(current_input, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence: remove first element, add prediction
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions

# Get the last LOOK_BACK days from the entire dataset
last_sequence = scaled_data[-LOOK_BACK:]
future_predictions = predict_future(model, last_sequence, scaler, PREDICTION_DAYS)

print(f"\nPredicted prices for next {PREDICTION_DAYS} days:")
print(future_predictions[:10].flatten())  # Show first 10 predictions

# ====================== Visualization ==================================
# Create future dates
last_date = pd.to_datetime(df['Date'].iloc[-1])
future_dates = [last_date + dt.timedelta(days=i+1) for i in range(PREDICTION_DAYS)]

# Plot historical data and predictions
plt.figure(figsize=(16, 6))

# Plot actual historical prices
historical_mid_prices = (df['High'] + df['Low']) / 2.0
plt.plot(pd.to_datetime(df['Date']), historical_mid_prices, label='Historical Price', color='blue', linewidth=1.5)

# Plot test predictions
test_dates = pd.to_datetime(df['Date'].iloc[train_size + LOOK_BACK:])
plt.plot(test_dates, test_predictions, label='Test Predictions', color='orange', linewidth=1.5, alpha=0.7)

# Plot future predictions
plt.plot(future_dates, future_predictions, label=f'{PREDICTION_DAYS}-Day Forecast', color='red', linewidth=2, linestyle='--')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.title(f'Stock Price Prediction - {PREDICTION_DAYS} Days Forecast', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Zoomed view - last 200 days + future predictions
plt.figure(figsize=(16, 6))
zoom_days = 200
zoom_start = max(0, len(df) - zoom_days)

plt.plot(pd.to_datetime(df['Date'].iloc[zoom_start:]), 
         historical_mid_prices.iloc[zoom_start:], 
         label='Historical Price', color='blue', linewidth=2)

plt.plot(future_dates, future_predictions, 
         label=f'{PREDICTION_DAYS}-Day Forecast', color='red', linewidth=2, linestyle='--', marker='o', markersize=3)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.title(f'Recent History and {PREDICTION_DAYS}-Day Forecast (Zoomed)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================== Save Model ==================================
model.save('lstm_stock_prediction_model.h5')
print("\nModel saved as 'lstm_stock_prediction_model.h5'")

# Save predictions to CSV
prediction_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions.flatten()
})
prediction_df.to_csv('future_predictions_90days.csv', index=False)
print("Predictions saved to 'future_predictions_90days.csv'")

# ====================== Summary Statistics ==================================
print("\n" + "="*60)
print("PREDICTION SUMMARY")
print("="*60)
print(f"Last known price: ${historical_mid_prices.iloc[-1]:.2f}")
print(f"Predicted price (Day 30): ${future_predictions[29][0]:.2f}")
print(f"Predicted price (Day 60): ${future_predictions[59][0]:.2f}")
print(f"Predicted price (Day 90): ${future_predictions[89][0]:.2f}")
print(f"Overall trend: {('Bullish ↑' if future_predictions[-1] > historical_mid_prices.iloc[-1] else 'Bearish ↓')}")
print(f"Price change: ${future_predictions[-1][0] - historical_mid_prices.iloc[-1]:.2f} " + 
      f"({((future_predictions[-1][0] / historical_mid_prices.iloc[-1] - 1) * 100):.2f}%)")
print("="*60)