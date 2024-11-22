import streamlit as st
import yfinance as yf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        st.write("Data is stationary")
    else:
        st.write("Data is not stationary")

# Streamlit app
st.title('BSE Sensex Time Series Analysis')

# Fetch data for BSE Sensex
symbol = "^BSESN"  # BSE Sensex symbol
data = yf.download(symbol, start="2010-01-01", end="2023-01-01")

# Ensure the index is in date format
data.index = pd.to_datetime(data.index)

# Check for missing values
st.write("Missing Values:")
st.write(data.isnull().sum())

# If there are missing values, you can fill them or drop them
data = data.dropna()

# Display the first few rows of the cleaned data
st.write("Cleaned Data:")
st.write(data.head())

# Plot the closing prices
st.write("Closing Prices:")
st.line_chart(data['Close'])

# Plot the distribution of closing prices
st.write("Distribution of Closing Prices:")
fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(data['Close'], kde=True, ax=ax)
ax.set_title('Distribution of BSE Sensex Closing Prices')
st.pyplot(fig)

# Check stationarity of the closing prices
st.write("Stationarity Check:")
check_stationarity(data['Close'])

# Differencing to make the data stationary
data['Close_diff'] = data['Close'].diff().dropna()

# Check stationarity of the differenced data
st.write("Stationarity Check after Differencing:")
check_stationarity(data['Close_diff'].dropna())

# Plot ACF and PACF to determine ARIMA parameters
st.write("ACF and PACF Plots:")
fig, ax = plt.subplots(2, 1, figsize=(14, 7))
plot_acf(data['Close_diff'].dropna(), ax=ax[0])
plot_pacf(data['Close_diff'].dropna(), ax=ax[1])
st.pyplot(fig)

# Fit the ARIMA model
model = ARIMA(data['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Summary of the model
st.write("ARIMA Model Summary:")
st.write(model_fit.summary())

# Forecast the next 30 days
forecast = model_fit.get_forecast(steps=30)
forecast_index = pd.date_range(data.index[-1], periods=30, freq='D')
forecast_values = forecast.predicted_mean

# Plot the forecast
st.write("Forecast:")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data['Close'], label='Historical')
ax.plot(forecast_index, forecast_values, label='Forecast')
ax.fill_between(forecast_index, forecast.conf_int()['lower Close'], forecast.conf_int()['upper Close'], alpha=0.2)
ax.set_title('BSE Sensex Closing Prices Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.legend()
st.pyplot(fig)

# Calculate MAE and RMSE
mae = mean_absolute_error(data['Close'][-30:], forecast_values)
rmse = mean_squared_error(data['Close'][-30:], forecast_values, squared=False)

st.write(f'MAE: {mae}')
st.write(f'RMSE: {rmse}')
