import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

def get_stock_predictions(ticker, start_date, end_date):
    # Loading historical stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    
    stock_data['Next_Close'] = stock_data['Close'].shift(-1)  # Shift the Close price by 1 day to predict the next day's Close price

    # Droping the last row as it will have NaN in 'Next_Close'
    stock_data = stock_data.dropna()

    # Defining features and target variable
    X = stock_data[['Open', 'High', 'Low', 'Close']]
    y = stock_data['Next_Close']

    # Spliting the data into training and testing sets while maintaining the index
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Building and training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    predictions = model.predict(X_test)

    # Creating a DataFrame to display the dates, actual, and predicted values
    results_df = pd.DataFrame({'Date': X_test.index, 'Actual': y_test, 'Predicted': predictions})
    results_df.set_index('Date', inplace=True)
    results_df.sort_index(inplace=True)  # Sort by date

    return results_df

# Streamlit UI
st.title('Stock Price Predictor')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')

# Date input for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2021-01-01'))

if st.button('Predict'):
    results_df = get_stock_predictions(ticker, start_date, end_date)
    st.write(f'## Predictions for {ticker}')
    st.write(results_df)