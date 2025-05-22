import os
import pandas as pd
import joblib
import warnings
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from umap import UMAP

warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

input_dir = "stock_features"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        ticker = file.split(".")[0]
        df = pd.read_csv(os.path.join(input_dir, file))

        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        df.dropna(inplace=True)

        feature_cols = [col for col in df.select_dtypes(include='number').columns if col != 'Target']
        X = df[feature_cols]
        y = df['Target']

        if len(X) < 100 or X.isnull().any().any():
            print(f"⚠️ Skipping {ticker} due to insufficient data or NaNs.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = UMAP(n_components=10, n_jobs=-1)  # Parallel computation
        X_reduced = reducer.fit_transform(X_scaled)

        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = 0

        for train_idx, test_idx in tscv.split(X_reduced):
            X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            clf = CalibratedClassifierCV(rf, cv=3)
            clf.fit(X_train_bal, y_train_bal)

            score = clf.score(X_test, y_test)
            if score > best_score:
                best_model = clf
                best_score = score

        # SHAP (optional sample for speed)
        explainer = shap.Explainer(best_model.calibrated_classifiers_[0].estimator, X_train_bal[:100])
        shap_values = explainer(X_train_bal[:100])
        shap.summary_plot(shap_values, X_train_bal[:100], show=False)

        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'reducer': reducer,
            'feature_names': feature_cols
        }, os.path.join(output_dir, f"{ticker}_model.pkl"))

        print(f"🤖 {ticker} model saved. Accuracy: {best_score:.2f}")
