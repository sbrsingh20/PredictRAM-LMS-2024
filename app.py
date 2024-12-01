import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load Data
def load_data(stock_file, economics_file):
    stock_data = pd.read_excel(stock_file)
    stock_data['Symbol'] = os.path.basename(stock_file).split('_')[0]  # Extract symbol from filename
    economics_data = pd.read_excel(economics_file)
    
    return stock_data, economics_data

# Preprocess Data
def preprocess_data(stock_data, economics_data):
    stock_data.columns = stock_data.columns.str.strip()
    economics_data.columns = economics_data.columns.str.strip()

    if 'Date' not in stock_data.columns:
        stock_data.rename(columns={'date': 'Date'}, inplace=True)

    if 'Date' not in economics_data.columns:
        economics_data.rename(columns={economics_data.columns[0]: 'Date'}, inplace=True)

    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    economics_data['Date'] = pd.to_datetime(economics_data['Date'], errors='coerce')

    stock_data = stock_data.dropna(subset=['Date'])
    economics_data = economics_data.dropna(subset=['Date'])

    # Merge data based on 'Date'
    merged_data = pd.merge(stock_data, economics_data, on='Date', how='inner')
    return merged_data

# Calculate Correlation
def calculate_correlation(stock_data, economic_event_data, event_column):
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna(subset=['Returns'])

    # Ensure the economic event data is aligned with the stock data
    economic_event_data = economic_event_data[['Date', event_column]]  # Selecting the column for the event
    economic_event_data['Date'] = pd.to_datetime(economic_event_data['Date'])
    
    # Merge the economic event data with stock data on 'Date'
    merged_event_data = pd.merge(stock_data[['Date', 'Returns']], economic_event_data, on='Date', how='inner')

    correlation = merged_event_data['Returns'].corr(merged_event_data[event_column])
    return correlation

# Financial Metrics Calculation
def calculate_financial_metrics(stock_returns, benchmark_returns):
    stock_returns = stock_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
    # Ensure the economic event data is aligned with the stock data
    aligned_returns = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    stock_returns = aligned_returns.iloc[:, 0]
    benchmark_returns = aligned_returns.iloc[:, 1]

    excess_returns = stock_returns - benchmark_returns
    mean_excess_returns = excess_returns.mean() * 252
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    sharpe_ratio = mean_excess_returns / benchmark_volatility
    benchmark_annualized_return = benchmark_returns.mean() * 252
    annualized_alpha = mean_excess_returns - benchmark_annualized_return

    treynor_ratio = mean_excess_returns / benchmark_volatility

    downside_returns = np.where(stock_returns < 0, stock_returns, 0)
    downside_deviation = np.std(downside_returns) * np.sqrt(252)
    sortino_ratio = mean_excess_returns / downside_deviation if downside_deviation != 0 else np.nan

    cumulative_returns = (1 + stock_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    r_squared = 1 - (np.var(stock_returns - benchmark_returns) / np.var(benchmark_returns))

    tracking_error = np.std(excess_returns) * np.sqrt(252)

    var_95 = np.percentile(excess_returns, 5)

    return {
        "Annualized Alpha (%)": annualized_alpha * 100,
        "Annualized Volatility (%)": benchmark_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Treynor Ratio": treynor_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown,
        "R-Squared": r_squared,
        "Annualized Tracking Error (%)": tracking_error * 100,
        "VaR (95%)": var_95
    }

# Train Machine Learning Model
def train_model(X, y, model_type="RandomForest"):
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    elif model_type == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError("Model type not supported")
    
    model.fit(X, y)
    return model

# Function to evaluate the model
def predict_and_evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2

# Predict close price based on economic event value
def predict_close_price(model, event_value, features, training_columns):
    # Prepare the feature set for prediction, ensuring the same columns as during training
    features_with_event = features.copy()
    
    # Add the new 'Economic Event' column, ensuring the correct column position
    features_with_event['Economic Event'] = event_value
    
    # Reorder the columns to match the order used during training
    features_with_event = features_with_event[training_columns]

    # Predict the close price based on the features
    prediction = model.predict(features_with_event)[-1]  # Predict based on the most recent row
    return prediction

# Main function
def main():
    st.title("Stock and Economic Data Analysis")

    # Set file paths for stock, economics, and benchmark files
    stock_folder = "stockdata"  # Folder containing your stock files
    economics_file = "IIP_data - Copy.xlsx"  # Your economic data file
    benchmark_file = "^NSEI.xlsx"  # Your benchmark index file
    
    # Load benchmark data
    benchmark_data, _ = load_data(benchmark_file, economics_file)
    benchmark_data = benchmark_data[['Date', 'Close']].dropna()
    benchmark_data['Returns'] = benchmark_data['Close'].pct_change().dropna()

    # Get stock symbols from the stockdata folder
    stock_files = [f for f in os.listdir(stock_folder) if f.endswith('.xlsx')]
    stock_symbols = [os.path.basename(f).split('_')[0] for f in stock_files]
    
    # Let user choose multiple stock symbols
    st.write("Available stock symbols:", stock_symbols)
    selected_symbols = st.multiselect("Select stock symbols to analyze", stock_symbols)

    if len(selected_symbols) == 0:
        st.error("No stock symbols selected. Please choose at least one symbol.")
        return

    st.write(f"Selected stock symbols for analysis: {selected_symbols}")

    results = []  # List to store results for each stock

    for stock_symbol in selected_symbols:
        # Find the stock file for the selected symbol
        selected_stock_file = None
        for stock_file in stock_files:
            if stock_symbol in stock_file:
                selected_stock_file = os.path.join(stock_folder, stock_file)
                break
        
        if selected_stock_file is None:
            st.error(f"Error: Stock file for '{stock_symbol}' not found.")
            continue

        st.write(f"\nProcessing {selected_stock_file}...")
        stock_data, economics_data = load_data(selected_stock_file, economics_file)
        merged_data = preprocess_data(stock_data, economics_data)

        economic_event_columns = list(economics_data.columns)
        
        # Adding the 'key' argument to ensure uniqueness
        economic_event = st.selectbox("Select an Economic Event", economic_event_columns, key=f"economic_event_{stock_symbol}")

        event_value = st.number_input(f"Enter the value for the economic event '{economic_event}'", value=0.0)

        stock_data_returns = merged_data['Close'].pct_change().dropna()

        # Train-Test Split
        features = merged_data[['Open', 'High', 'Low', 'Volume']]
        target = merged_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Store the training feature columns to use for prediction
        training_columns = features.columns.tolist()

        # Store results and model parameters
        model_results = []

        # List of models to evaluate
        model_types = ["RandomForest", "XGBoost", "LightGBM", "LinearRegression"]

        for model_type in model_types:
            st.write(f"\nTraining model: {model_type}")
            model = train_model(X_train, y_train, model_type)
            predictions, mse, r2 = predict_and_evaluate(model, X_test, y_test)
            model_results.append({
                "Model": model_type,
                "Mean Squared Error": mse,
                "R-Squared": r2,
                "Parameters": model.get_params()
            })
            st.write(f"{model_type} - MSE: {mse:.2f}, R-Squared: {r2:.2f}")

            # Predict close price based on the economic event value
            predicted_close_price = predict_close_price(model, event_value, features, training_columns)
            st.write(f"Predicted Close Price for {stock_symbol} based on event value: {predicted_close_price:.2f}")

    # Show final results
    for result in model_results:
        st.write(f"\n{result['Model']} - MSE: {result['Mean Squared Error']:.2f}, R-Squared: {result['R-Squared']:.2f}")

if __name__ == "__main__":
    main()
