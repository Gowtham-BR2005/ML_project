from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
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
    
    # Indian Stock: Reliance Industries on BSE
    ticker = "RELIANCE.BSE"  # Reliance on Bombay Stock Exchange
    # Alternative: "RELIANCE.NS" for NSE (National Stock Exchange)
    
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
    df = pd.read_csv('stock_market_data-RELIANCE.BSE.csv')

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

# ====================== Data Preparation (IMPROVED) ==================================
# Use multiple features instead of just mid price
features_to_use = ['Open', 'High', 'Low', 'Close']
data_features = df[features_to_use].values

# Calculate additional technical indicators
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Volatility'] = df['Close'].rolling(window=10).std()
df['Price_Change'] = df['Close'].pct_change()

# Drop NaN values from rolling calculations
df = df.dropna().reset_index(drop=True)

# Use Close price as target (but train with multiple features)
close_prices = df['Close'].values.reshape(-1, 1)
all_features = df[['Open', 'High', 'Low', 'Close', 'MA_5', 'MA_20', 'Volatility']].values

print(f"Dataset size after cleaning: {len(df)}")

# Normalize the data - separate scalers for features and target
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler_features.fit_transform(all_features)
scaled_target = scaler_target.fit_transform(close_prices)

# Split into train and test (80/20 split)
train_size = int(len(scaled_features) * 0.8)
train_features = scaled_features[:train_size]
test_features = scaled_features[train_size:]
train_target = scaled_target[:train_size]
test_target = scaled_target[train_size:]

print(f"Training samples: {len(train_features)}")
print(f"Testing samples: {len(test_features)}")

# ====================== Create Sequences (IMPROVED) ==================================
def create_sequences(features, target, look_back=60):
    """
    Create sequences with multiple features
    """
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i-look_back:i])  # Multiple features
        y.append(target[i, 0])  # Single target (Close price)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_features, train_target, LOOK_BACK)
X_test, y_test = create_sequences(test_features, test_target, LOOK_BACK)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ====================== Build IMPROVED LSTM Model ==================================
def build_improved_lstm_model(input_shape):
    model = Sequential([
        # Bidirectional LSTM layers capture patterns in both directions
        Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.3),
        
        LSTM(units=64, return_sequences=False),
        Dropout(0.3),
        
        Dense(units=32, activation='relu'),
        Dropout(0.2),
        
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    # Use a lower learning rate for more stable training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    return model

model = build_improved_lstm_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# ====================== Train Model (IMPROVED) ==================================
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)

print("\nTraining model... This may take 5-15 minutes depending on your hardware.")
print("The model needs 50-100 epochs to learn properly. Please be patient!")

history = model.fit(
    X_train, y_train,
    batch_size=64,  # Larger batch size for stability
    epochs=100,  # More epochs
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====================== Evaluate Model ==================================
# Make predictions on test set
test_predictions = model.predict(X_test)
test_predictions = scaler_target.inverse_transform(test_predictions)
y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = np.mean((test_predictions - y_test_actual)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_predictions - y_test_actual))
mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100

print(f"\nTest Set Performance:")
print(f"RMSE: ₹{rmse:.2f}")
print(f"MAE: ₹{mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Visualize training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model MAE')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_test_actual, test_predictions, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================== Predict Future Days (IMPROVED) ==================================
def predict_future_improved(model, last_features, scaler_features, scaler_target, days=90):
    """
    Predict future stock prices using the last known features
    For future predictions, we'll use a more conservative approach
    """
    predictions = []
    current_features = last_features.copy()
    
    for day in range(days):
        # Reshape for prediction
        current_input = current_features.reshape(1, LOOK_BACK, -1)
        
        # Predict next day
        next_pred_scaled = model.predict(current_input, verbose=0)
        predictions.append(next_pred_scaled[0, 0])
        
        # Create new feature row for next prediction
        # Use predicted close price and estimated other features
        last_close_scaled = next_pred_scaled[0, 0]
        
        # Estimate other features based on last close
        # This is a simple approach - in production you'd want more sophisticated methods
        estimated_open = last_close_scaled
        estimated_high = last_close_scaled * 1.005  # Slight variation
        estimated_low = last_close_scaled * 0.995
        
        # Calculate moving averages from prediction history
        recent_preds = predictions[-20:] if len(predictions) >= 20 else predictions
        ma_5 = np.mean(predictions[-5:]) if len(predictions) >= 5 else last_close_scaled
        ma_20 = np.mean(recent_preds)
        volatility = np.std(recent_preds) if len(recent_preds) > 1 else 0
        
        new_feature_row = np.array([estimated_open, estimated_high, estimated_low, 
                                     last_close_scaled, ma_5, ma_20, volatility])
        
        # Update sequence: remove first row, add new prediction
        current_features = np.vstack([current_features[1:], new_feature_row])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler_target.inverse_transform(predictions)
    
    return predictions

# Get the last LOOK_BACK days from the entire dataset
last_features = scaled_features[-LOOK_BACK:]
future_predictions = predict_future_improved(model, last_features, scaler_features, 
                                             scaler_target, PREDICTION_DAYS)

print(f"\nPredicted prices for next {PREDICTION_DAYS} days:")
print(future_predictions[:min(10, PREDICTION_DAYS)].flatten())

# ====================== Visualization ==================================
# Create future dates
last_date = pd.to_datetime(df['Date'].iloc[-1])
future_dates = [last_date + dt.timedelta(days=i+1) for i in range(PREDICTION_DAYS)]

# Plot historical data and predictions
plt.figure(figsize=(18, 7))

# Plot actual historical prices
historical_prices = df['Close']
plt.plot(pd.to_datetime(df['Date']), historical_prices, label='Historical Price', 
         color='blue', linewidth=1.5, alpha=0.8)

# Plot test predictions
test_dates = pd.to_datetime(df['Date'].iloc[train_size + LOOK_BACK:])
plt.plot(test_dates, test_predictions, label='Test Predictions', 
         color='green', linewidth=1.5, alpha=0.7)

# Plot actual test values for comparison
plt.plot(test_dates, y_test_actual, label='Actual Test Values', 
         color='cyan', linewidth=1, alpha=0.6, linestyle=':')

# Plot future predictions
plt.plot(future_dates, future_predictions, label=f'{PREDICTION_DAYS}-Day Forecast', 
         color='red', linewidth=2, linestyle='--', marker='o', markersize=4)

# Add confidence band for future predictions (simple approach)
std_dev = np.std(test_predictions - y_test_actual)
upper_bound = future_predictions + 2 * std_dev
lower_bound = future_predictions - 2 * std_dev
plt.fill_between(future_dates, lower_bound.flatten(), upper_bound.flatten(), 
                 color='red', alpha=0.1, label='Confidence Interval')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (₹)', fontsize=12)
plt.title(f'Reliance Stock Price Prediction - {PREDICTION_DAYS} Days Forecast (MAPE: {mape:.2f}%)', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Zoomed view - last 200 days + future predictions
plt.figure(figsize=(18, 7))
zoom_days = 200
zoom_start = max(0, len(df) - zoom_days)

plt.plot(pd.to_datetime(df['Date'].iloc[zoom_start:]), 
         historical_prices.iloc[zoom_start:], 
         label='Historical Price', color='blue', linewidth=2)

plt.plot(future_dates, future_predictions, 
         label=f'{PREDICTION_DAYS}-Day Forecast', color='red', 
         linewidth=2.5, linestyle='--', marker='o', markersize=4)

# Add confidence band
plt.fill_between(future_dates, lower_bound.flatten(), upper_bound.flatten(), 
                 color='red', alpha=0.15, label='Confidence Interval (±2σ)')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (₹)', fontsize=12)
plt.title(f'Reliance - Recent History and {PREDICTION_DAYS}-Day Forecast (Zoomed)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================== Save Model ==================================
model.save('lstm_stock_prediction_model.h5')
print("\nModel saved as 'lstm_stock_prediction_model.h5'")

# Save predictions to CSV
prediction_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions.flatten(),
    'Lower_Bound': lower_bound.flatten(),
    'Upper_Bound': upper_bound.flatten()
})
prediction_df.to_csv(f'future_predictions_{PREDICTION_DAYS}days.csv', index=False)
print(f"Predictions saved to 'future_predictions_{PREDICTION_DAYS}days.csv'")

# ====================== Summary Statistics ==================================
print("\n" + "="*70)
print("RELIANCE STOCK PREDICTION SUMMARY")
print("="*70)
print(f"Model Accuracy (MAPE): {mape:.2f}%")
print(f"Last known price: ₹{historical_prices.iloc[-1]:.2f}")

# Show predictions at different intervals
if PREDICTION_DAYS >= 1:
    print(f"Predicted price (Day 1): ₹{future_predictions[0][0]:.2f}")
if PREDICTION_DAYS >= 7:
    print(f"Predicted price (Day 7): ₹{future_predictions[6][0]:.2f}")
if PREDICTION_DAYS >= 30:
    print(f"Predicted price (Day 30): ₹{future_predictions[29][0]:.2f}")
if PREDICTION_DAYS >= 90:
    print(f"Predicted price (Day 90): ₹{future_predictions[89][0]:.2f}")

final_price = future_predictions[-1][0]
last_price = historical_prices.iloc[-1]
print(f"Predicted price (Day {PREDICTION_DAYS}): ₹{final_price:.2f}")
print(f"\nOverall trend: {('Bullish ↑' if final_price > last_price else 'Bearish ↓')}")
print(f"Price change: ₹{final_price - last_price:.2f} " + 
      f"({((final_price / last_price - 1) * 100):.2f}%)")
print("="*70)
print("\nNote: Longer-term predictions (30-90 days) are less reliable than short-term.")
print("Use these predictions as one of many tools for decision-making.")