import yfinance as yf
import pandas as pd
import os

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    return data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    os.makedirs("stock_data", exist_ok=True)

    for ticker in tickers:
        df = get_stock_data(ticker, "2020-01-01", "2024-12-31")
        df.to_csv(f"stock_data/{ticker}.csv", index=False)
        print(f"📅 Saved data for {ticker}")
