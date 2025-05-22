import os
import pandas as pd
import joblib

feature_dir = "stock_features"
model_dir = "models"
results = {}
cost_per_trade = 0.001  # 0.1%

for file in os.listdir(feature_dir):
    if file.endswith(".csv"):
        ticker = file.split(".")[0]
        df = pd.read_csv(os.path.join(feature_dir, file))
        model_path = os.path.join(model_dir, f"{ticker}_model.pkl")

        if not os.path.exists(model_path):
            print(f"⚠️ Model for {ticker} not found, skipping.")
            continue

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        reducer = model_data['reducer']
        features = df[model_data['feature_names']].copy()

        X_scaled = scaler.transform(features)
        X_reduced = reducer.transform(X_scaled)

        df['Prediction'] = model.predict(X_reduced)
        df['Prob_Up'] = model.predict_proba(X_reduced)[:, 1]
        df['Strategy_Return'] = df['Close'].pct_change().shift(-1) * df['Prediction']
        df['Strategy_Return'] -= cost_per_trade * (df['Prediction'].diff().fillna(0) != 0).astype(int)
        df['Cumulative_Return'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()

        last_prob = df['Prob_Up'].iloc[-1]
        results[ticker] = last_prob

        df.to_csv(f"backtest_{ticker}.csv", index=False)
        print(f"📈 Backtested {ticker}, Prob of up: {last_prob:.2%}")

if results:
    best_stock = max(results.items(), key=lambda x: x[1])
    print(f"\n💡 Best stock to invest in next: {best_stock[0]} with probability {best_stock[1]:.2%}")
