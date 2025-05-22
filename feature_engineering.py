import pandas as pd
import os
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def add_indicators(df):
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Close', 'High', 'Low', 'Volume'], inplace=True)

    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['MACD'] = MACD(close=df['Close']).macd_diff()
    df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
    df['Bollinger_Width'] = BollingerBands(close=df['Close']).bollinger_wband()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['ROC'] = ROCIndicator(close=df['Close']).roc()

    return df.dropna()

if __name__ == "__main__":
    input_dir = "stock_data"
    output_dir = "stock_features"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            ticker = file.split(".")[0]
            df = pd.read_csv(os.path.join(input_dir, file))
            df = add_indicators(df)
            df.to_csv(os.path.join(output_dir, f"{ticker}.csv"), index=False)
            print(f"🛠️ Features saved for {ticker}")
