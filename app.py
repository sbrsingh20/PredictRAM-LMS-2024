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
    stock_data['Symbol'] = os.path.basename(stock_file.name).split('_')[0]  # Extract symbol from filename
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

    merged_data = pd.merge(stock_data, economics_data, on='Date', how='inner')
    return merged_data

# Calculate Correlation
def calculate_correlation(stock_data, economic_event_data):
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna(subset=['Returns'])
    
    # Ensure the economic event data is aligned with the stock data
    economic_event_data = economic_event_data.loc[stock_data.index]
    correlation = stock_data['Returns'].corr(economic_event_data)
    return correlation

# Financial Metrics Calculation
def calculate_financial_metrics(stock_returns, benchmark_returns):
    stock_returns = stock_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    
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

# Main function
def main():
    st.title("Stock and Economic Data Analysis")

    # Upload files
    stock_file = st.file_uploader("Upload Stock Data File", type="xlsx")
    economics_file = st.file_uploader("Upload Economic Data File", type="xlsx")
    benchmark_file = st.file_uploader("Upload Benchmark File (e.g., ^NSEI.xlsx)", type="xlsx")
    
    if stock_file is not None and economics_file is not None and benchmark_file is not None:
        # Load data
        benchmark_data, _ = load_data(benchmark_file, economics_file)
        benchmark_data = benchmark_data[['Date', 'Close']].dropna()
        benchmark_data['Returns'] = benchmark_data['Close'].pct_change().dropna()

        # Load stock data
        stock_data, economics_data = load_data(stock_file, economics_file)
        merged_data = preprocess_data(stock_data, economics_data)

        economic_event_columns = list(economics_data.columns)
        economic_event = st.selectbox("Select Economic Event", economic_event_columns)

        event_value = st.number_input(f"Enter the value for the economic event '{economic_event}'", value=0.0)

        stock_data_returns = merged_data['Close'].pct_change().dropna()

        # Train-Test Split
        features = merged_data[['Open', 'High', 'Low', 'Volume']]
        target = merged_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # List of models to evaluate
        model_types = ["RandomForest", "XGBoost", "LightGBM", "LinearRegression"]
        model_results = []

        for model_type in model_types:
            st.write(f"\nTraining model: {model_type}")
            model = train_model(X_train, y_train, model_type)
            predictions, mse, r2 = predict_and_evaluate(model, X_test, y_test)

            model_info = {
                "Model": model_type,
                "Mean Squared Error": mse,
                "R-Squared": r2,
                "Parameters": model.get_params() if hasattr(model, 'get_params') else 'N/A'
            }
            model_results.append(model_info)

            # Display predicted closing price for the latest day
            last_row = merged_data.iloc[-1]
            user_input = {
                'Open': last_row['Open'],
                'High': last_row['High'],
                'Low': last_row['Low'],
                'Volume': last_row['Volume']
            }
            user_df = pd.DataFrame([user_input])
            predicted_close = model.predict(user_df)

            st.write(f"Predicted Closing Price using {model_type}: {predicted_close[0]:.4f}")

        # Calculate and display correlation
        correlation = calculate_correlation(merged_data, merged_data[economic_event])
        st.write(f"Correlation between stock returns and economic event: {correlation:.4f}")

        # Financial metrics calculation
        financial_metrics = calculate_financial_metrics(stock_data_returns, benchmark_data['Returns'].dropna())
        financial_metrics_df = pd.DataFrame([financial_metrics])
        st.table(financial_metrics_df)

# Run the app
if __name__ == "__main__":
    main()
