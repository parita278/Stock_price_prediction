# FUTURE DAY PREDICTION

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Stock Price Prediction (Live LSTM)", layout="centered")
st.title("📈 Live Stock Price Prediction with LSTM")

ticker = st.text_input('Enter stock ticker')

lookback_days=60
prediction_days=3
if st.button('Predict stock price'):
    today = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start='2024-01-01', end=today)
    st.dataframe(df.tail())

    data=df[['Close']].copy()
    data=data.dropna()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split data into train and test
    train_size=int(len(scaled_data)*0.8)
    train_data=scaled_data[:train_size]
    test_data=scaled_data[train_size-lookback_days:]

    # Prepare training data
    def create_sequences(data,lookback_days):
        X,y=[],[]
        for i in range(lookback_days,len(data)):
            X.append(data[i-lookback_days:i,0])
            y.append(data[i,0])
        return np.array(X),np.array(y)
    X_train,y_train=create_sequences(train_data,lookback_days)
    X_test,y_test=create_sequences(test_data,lookback_days)

    # Reshape for LSTM [samples, time steps, features]
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train_lstm = np.reshape(y_train, (y_train.shape[0], 1))
    y_test_lstm = np.reshape(y_test, (y_test.shape[0], 1)) 

    # --- LSTM Model ---
    lstm_model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(lookback_days, 1)),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100),
        Dropout(0.2),
        Dense(units=1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    with st.spinner('Training model...'):
        history_lstm = lstm_model.fit(
            X_train_lstm, y_train_lstm,
            batch_size=32,
            epochs=20,
            validation_data=(X_test_lstm, y_test_lstm),
            callbacks=[early_stop_lstm],
            verbose=1
        )

    # Make predictions on test data for LSTM
    lstm_predictions=lstm_model.predict(X_test_lstm)

    # Inverse transform to get actual prices for LSTM
    lstm_predictions_actual=scaler.inverse_transform(lstm_predictions)
    y_test_actual_lstm=scaler.inverse_transform(y_test_lstm)

    # Calculate accuracy metrics for LSTM
    mse_lstm = mean_squared_error(y_test_actual_lstm, lstm_predictions_actual)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(y_test_actual_lstm, lstm_predictions_actual)
    r2_lstm = r2_score(y_test_actual_lstm, lstm_predictions_actual)

    # Calculate percentage accuracy (custom metric for stock prediction) for LSTM
    percentage_errors_lstm = np.abs((y_test_actual_lstm - lstm_predictions_actual) / y_test_actual_lstm) * 100
    accuracy_percentage_lstm = 100 - np.mean(percentage_errors_lstm)

    st.subheader("LSTM MODEL ACCURACY METRICS")
    st.markdown(f"Mean Squared Error (MSE):{mse_lstm:.4f}")
    st.markdown(f"Root Mean Squared Error (RMSE):{rmse_lstm:.4f}")
    st.markdown(f"Mean Absolute Error (MAE):{mae_lstm:.4f}")
    st.markdown(f"R² Score:{r2_lstm:.4f}")
    st.markdown(f"Directional Accuracy:{accuracy_percentage_lstm:.2f}%")


    # --- XGBoost Model ---
    X_train_xgb = X_train.reshape(X_train.shape[0], -1)
    X_test_xgb = X_test.reshape(X_test.shape[0], -1)
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)

    with st.spinner('Training the XGBOOST model...'):
        xgb_model.fit(X_train_xgb, y_train,
                  eval_set=[(X_test_xgb, y_test)],
                  verbose=False)
        
    # Make predictions on test data for XGBoost
    xgb_predictions = xgb_model.predict(X_test_xgb)

    # Inverse transform to get actual prices for XGBoost
    xgb_predictions_actual = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))
    y_test_actual_xgb = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate accuracy metrics for XGBoost
    mse_xgb = mean_squared_error(y_test_actual_xgb, xgb_predictions_actual)
    rmse_xgb = np.sqrt(mse_xgb)
    mae_xgb = mean_absolute_error(y_test_actual_xgb, xgb_predictions_actual)
    r2_xgb = r2_score(y_test_actual_xgb, xgb_predictions_actual)

    # Calculate percentage accuracy (custom metric for stock prediction) for XGBoost
    percentage_errors_xgb = np.abs((y_test_actual_xgb - xgb_predictions_actual) / y_test_actual_xgb) * 100
    accuracy_percentage_xgb = 100 - np.mean(percentage_errors_xgb)

    st.subheader("XGBOOST MODEL ACCURACY METRICS")
    st.markdown(f"Mean Squared Error (MSE):{mse_xgb:.4f}")
    st.markdown(f"Root Mean Squared Error (RMSE):{rmse_xgb:.4f}")
    st.markdown(f"Mean Absolute Error (MAE):{mae_xgb:.4f}")
    st.markdown(f"R² Score:{r2_xgb:.4f}")
    st.markdown(f"Directional Accuracy:{accuracy_percentage_xgb:.2f}%")
    st.markdown("--------------------------------------------------------------")

    # Future predictions (Using LSTM for now as it was in the original code)
    st.markdown(f"\nGenerating {prediction_days} days of future predictions using LSTM...")

    # Get last sequence for future prediction
    last_sequence_lstm = scaled_data[-lookback_days:]
    future_predictions_lstm = []

    for i in range(prediction_days):
        # Reshape for prediction
        current_sequence_lstm = last_sequence_lstm.reshape((1, lookback_days, 1))

        # Predict next value
        next_pred_lstm = lstm_model.predict(current_sequence_lstm, verbose=0)
        future_predictions_lstm.append(next_pred_lstm[0, 0])

        # Update sequence for next prediction
        last_sequence_lstm = np.append(last_sequence_lstm[1:], next_pred_lstm, axis=0)

    # Inverse transform future predictions
    future_predictions_lstm = np.array(future_predictions_lstm).reshape(-1, 1)
    future_predictions_lstm = scaler.inverse_transform(future_predictions_lstm)

    # Create future dates
    last_date = data.index[-1]
    future_dates = []
    for i in range(1, prediction_days + 1):
        future_date = last_date + timedelta(days=i)
        future_dates.append(future_date.strftime('%Y-%m-%d'))

    # Create future predictions dataframe
    future_df_lstm = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price_LSTM': future_predictions_lstm.flatten()
    })
    st.subheader("Future Predictions (LSTM)")
    st.dataframe(future_df_lstm)


    # Create results dictionary
    results = {
        'lstm_accuracy_metrics': {
            'mse': float(mse_lstm),
            'rmse': float(rmse_lstm),
            'mae': float(mae_lstm),
            'r2_score': float(r2_lstm),
            'percentage_accuracy': float(accuracy_percentage_lstm)
        },
         'xgb_accuracy_metrics': {
            'mse': float(mse_xgb),
            'rmse': float(rmse_xgb),
            'mae': float(mae_xgb),
            'r2_score': float(r2_xgb),
            'percentage_accuracy': float(accuracy_percentage_xgb)
        },
        'lstm_future_predictions': future_df_lstm.to_dict('records'),
        'model_performance': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'lstm_epochs_trained': len(history_lstm.history['loss']),

        }
    }

    st.markdown(f"\nLSTM Model Accuracy: {results['lstm_accuracy_metrics']['percentage_accuracy']:.2f}%")
    st.markdown(f"LSTM R² Score: {results['lstm_accuracy_metrics']['r2_score']:.4f}")
    st.markdown(f"\nXGBoost Model Accuracy: {results['xgb_accuracy_metrics']['percentage_accuracy']:.2f}%")
    st.markdown(f"XGBoost R² Score: {results['xgb_accuracy_metrics']['r2_score']:.4f}")

    # Plot
    future_dates = pd.to_datetime(future_dates)
    test_dates = df.index[-len(y_test_actual_lstm):]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
    ax.plot(test_dates, lstm_predictions_actual, label='Predicted Prices (Test)', color='orange')
    ax.plot(future_dates, future_predictions_lstm, label='Predicted Prices (Future)', color='green')
    ax.set_title(f"{ticker} Price Prediction (Next 10 Days)")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)






# # FUTURE MINITES PREDICTION

# import pandas as pd
# import numpy as np
# import streamlit as st
# import yfinance as yf
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# import time

# st.set_page_config(page_title="Stock Price Prediction (Live LSTM)", layout="centered")
# st.title("📈 Live Stock Price Prediction with LSTM")

# ticker = st.text_input('Enter stock ticker')

# lookback_days=60
# prediction_days=3
# if st.button('Predict stock price'):
#     today = datetime.today().strftime('%Y-%m-%d')
#     df = yf.download(ticker, period="2d", interval="1m")
#     st.dataframe(df.tail())

#     data=df[['Close']].copy()
#     data=data.dropna()

#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)

#     # Split data into train and test
#     train_size=int(len(scaled_data)*0.8)
#     train_data=scaled_data[:train_size]
#     test_data=scaled_data[train_size-lookback_days:]

#     # Prepare training data
#     def create_sequences(data,lookback_days):
#         X,y=[],[]
#         for i in range(lookback_days,len(data)):
#             X.append(data[i-lookback_days:i,0])
#             y.append(data[i,0])
#         return np.array(X),np.array(y)
#     X_train,y_train=create_sequences(train_data,lookback_days)
#     X_test,y_test=create_sequences(test_data,lookback_days)

#     # Reshape for LSTM [samples, time steps, features]
#     X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#     X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
#     y_train_lstm = y_train.reshape((y_train.shape[0], 1))
#     y_test_lstm = y_test.reshape((y_test.shape[0], 1)) 

#     # --- LSTM Model ---
#     lstm_model = Sequential([
#         LSTM(units=100, return_sequences=True, input_shape=(lookback_days, 1)),
#         Dropout(0.2),
#         LSTM(units=100, return_sequences=True),
#         Dropout(0.2),
#         LSTM(units=100),
#         Dropout(0.2),
#         Dense(units=1)
#     ])
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#     early_stop_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     with st.spinner('Training model...'):
#         history_lstm = lstm_model.fit(
#             X_train_lstm, y_train_lstm,
#             batch_size=32,
#             epochs=20,
#             validation_data=(X_test_lstm, y_test_lstm),
#             callbacks=[early_stop_lstm],
#             verbose=1
#         )

#     # Make predictions on test data for LSTM
#     lstm_predictions=lstm_model.predict(X_test_lstm)

#     # Inverse transform to get actual prices for LSTM
#     lstm_predictions_actual=scaler.inverse_transform(lstm_predictions)
#     y_test_actual_lstm=scaler.inverse_transform(y_test_lstm)

#     # Calculate accuracy metrics for LSTM
#     mse_lstm = mean_squared_error(y_test_actual_lstm, lstm_predictions_actual)
#     rmse_lstm = np.sqrt(mse_lstm)
#     mae_lstm = mean_absolute_error(y_test_actual_lstm, lstm_predictions_actual)
#     r2_lstm = r2_score(y_test_actual_lstm, lstm_predictions_actual)

#     # Calculate percentage accuracy (custom metric for stock prediction) for LSTM
#     percentage_errors_lstm = np.abs((y_test_actual_lstm - lstm_predictions_actual) / y_test_actual_lstm) * 100
#     accuracy_percentage_lstm = 100 - np.mean(percentage_errors_lstm)

#     st.subheader("LSTM MODEL ACCURACY METRICS")
#     st.markdown(f"Mean Squared Error (MSE):{mse_lstm:.4f}")
#     st.markdown(f"Root Mean Squared Error (RMSE):{rmse_lstm:.4f}")
#     st.markdown(f"Mean Absolute Error (MAE):{mae_lstm:.4f}")
#     st.markdown(f"R² Score:{r2_lstm:.4f}")
#     st.markdown(f"Directional Accuracy:{accuracy_percentage_lstm:.2f}%")


#     # --- XGBoost Model ---
#     X_train_xgb = X_train.reshape(X_train.shape[0], -1)
#     X_test_xgb = X_test.reshape(X_test.shape[0], -1)
#     xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)

#     with st.spinner('Training the XGBOOST model...'):
#         xgb_model.fit(X_train_xgb, y_train,
#                   eval_set=[(X_test_xgb, y_test)],
#                   verbose=False)
        
#     # Make predictions on test data for XGBoost
#     xgb_predictions = xgb_model.predict(X_test_xgb)

#     # Inverse transform to get actual prices for XGBoost
#     xgb_predictions_actual = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))
#     y_test_actual_xgb = scaler.inverse_transform(y_test.reshape(-1, 1))

#     # Calculate accuracy metrics for XGBoost
#     mse_xgb = mean_squared_error(y_test_actual_xgb, xgb_predictions_actual)
#     rmse_xgb = np.sqrt(mse_xgb)
#     mae_xgb = mean_absolute_error(y_test_actual_xgb, xgb_predictions_actual)
#     r2_xgb = r2_score(y_test_actual_xgb, xgb_predictions_actual)

#     # Calculate percentage accuracy (custom metric for stock prediction) for XGBoost
#     percentage_errors_xgb = np.abs((y_test_actual_xgb - xgb_predictions_actual) / y_test_actual_xgb) * 100
#     accuracy_percentage_xgb = 100 - np.mean(percentage_errors_xgb)

#     st.subheader("XGBOOST MODEL ACCURACY METRICS")
#     st.markdown(f"Mean Squared Error (MSE):{mse_xgb:.4f}")
#     st.markdown(f"Root Mean Squared Error (RMSE):{rmse_xgb:.4f}")
#     st.markdown(f"Mean Absolute Error (MAE):{mae_xgb:.4f}")
#     st.markdown(f"R² Score:{r2_xgb:.4f}")
#     st.markdown(f"Directional Accuracy:{accuracy_percentage_xgb:.2f}%")
#     st.markdown("--------------------------------------------------------------")

#     # Future predictions (Using LSTM for now as it was in the original code)
#     # st.markdown(f"\nGenerating {prediction_days} days of future predictions using LSTM...")

#     # # Get last sequence for future prediction
#     # last_sequence_lstm = scaled_data[-lookback_days:]
#     # future_predictions_lstm = []

#     # for i in range():
#     #     # Reshape for prediction
#     #     current_sequence_lstm = last_sequence_lstm.reshape((1, lookback_days, 1))

#     #     # Predict next value
#     #     next_pred_lstm = lstm_model.predict(current_sequence_lstm, verbose=0)
#     #     future_predictions_lstm.append(next_pred_lstm[0, 0])

#     #     # Update sequence for next prediction
#     #     last_sequence_lstm = np.append(last_sequence_lstm[1:], next_pred_lstm, axis=0)

#     # # Inverse transform future predictions
#     # future_predictions_lstm = np.array(future_predictions_lstm).reshape(-1, 1)
#     # future_predictions_lstm = scaler.inverse_transform(future_predictions_lstm)

#     # # Create future dates
#     # # last_date = data.index[-1]
#     # # future_dates = []
#     # # for i in range(1, prediction_days + 1):
#     # #     future_date = last_date + timedelta(minutes=i)
#     # #     future_dates.append(future_date.strftime('%H:%M:%S'))

#     # # Create future predictions dataframe
#     # future_df_lstm = pd.DataFrame({
#     #     # 'time': future_dates,
#     #     'Predicted_Price_LSTM': future_predictions_lstm.flatten()
#     # })
#     # st.subheader("Future Predictions (LSTM)")
#     # st.dataframe(future_df_lstm)

#     # After training LSTM and scaling data
#     st.subheader("Live 1-Minute Ahead Predictions (LSTM)")

#     # Get last sequence for prediction
#     last_sequence = scaled_data[-lookback_days:].copy()
#     placeholder = st.empty()

#     start_time="9:15:00"
#     end_time="3:30:00"
#     for i in range(360):  # Example: predict 5 future minutes in live mode
#         now = datetime.now().strftime("%H:%M:%S")
#         if now>=start_time or now<=end_time:
#             # Predict next minute price
#             current_sequence = last_sequence.reshape((1, lookback_days, 1))
#             next_pred_scaled = lstm_model.predict(current_sequence, verbose=0)
#             next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]

#             # Show prediction
#             placeholder.markdown(f"**[{now}] Predicted next minute price:** {next_pred:.2f}")

#             # Update last sequence with new predicted value
#             last_sequence = np.append(last_sequence[1:], next_pred_scaled, axis=0)

#             # Wait 1 minute
#             time.sleep(60) 

#         else:
#             st.markdown("market is close")
#             break



#     # Create results dictionary
#     results = {
#         'lstm_accuracy_metrics': {
#             'mse': float(mse_lstm),
#             'rmse': float(rmse_lstm),
#             'mae': float(mae_lstm),
#             'r2_score': float(r2_lstm),
#             'percentage_accuracy': float(accuracy_percentage_lstm)
#         },
#          'xgb_accuracy_metrics': {
#             'mse': float(mse_xgb),
#             'rmse': float(rmse_xgb),
#             'mae': float(mae_xgb),
#             'r2_score': float(r2_xgb),
#             'percentage_accuracy': float(accuracy_percentage_xgb)
#         },
#         # 'lstm_future_predictions': future_df_lstm.to_dict('records'),
#         # 'model_performance': {
#         #     'training_samples': len(X_train),
#         #     'test_samples': len(X_test),
#         #     'lstm_epochs_trained': len(history_lstm.history['loss']),

#         # }
#     }

#     st.markdown(f"\nLSTM Model Accuracy: {results['lstm_accuracy_metrics']['percentage_accuracy']:.2f}%")
#     st.markdown(f"LSTM R² Score: {results['lstm_accuracy_metrics']['r2_score']:.4f}")
#     st.markdown(f"\nXGBoost Model Accuracy: {results['xgb_accuracy_metrics']['percentage_accuracy']:.2f}%")
#     st.markdown(f"XGBoost R² Score: {results['xgb_accuracy_metrics']['r2_score']:.4f}")

#     # Plot
#     # future_dates = pd.to_datetime(future_dates)
#     test_dates = df.index[-len(y_test_actual_lstm):]
#     fig, ax = plt.subplots(figsize=(20, 6))
#     ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
#     ax.plot(test_dates, lstm_predictions_actual, label='Predicted Prices (Test)', color='orange')
#     # ax.plot(future_dates, future_predictions_lstm, label='Predicted Prices (Future)', color='green')
#     ax.set_title(f"{ticker} Price Prediction (Next 3 minites)")
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.legend()
#     st.pyplot(fig)








