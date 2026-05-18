import os
import sys
import warnings
import pickle

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta, date

app = Flask(__name__)

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))
DATA_DIR  = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))

print(f"MODEL_DIR : {MODEL_DIR}")
print(f"DATA_DIR  : {DATA_DIR}")

# ── Feature names (must match train_models.py) ────────────────────────────
FEATURE_NAMES = [
    'day', 'month', 'year', 'dayofweek',
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7',  'rolling_std_14',  'rolling_std_30',
]

# ── Global dataset cache ──────────────────────────────────────────────────
_historical_data = None

def load_historical_data():
    global _historical_data
    if _historical_data is not None:
        return _historical_data
    csv_path = os.path.join(DATA_DIR, 'usd_idr_daily.csv')
    if not os.path.exists(csv_path):
        print(f"Dataset not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
    df = df.sort_values('Date').reset_index(drop=True)
    _historical_data = df
    print(f"Loaded {len(df)} rows  [{df['Date'].min()} → {df['Date'].max()}]")
    return df

# ── Model holders ─────────────────────────────────────────────────────────
scaler = lr_model = ann_model = lstm_model = kmeans_model = bp_model = all_metrics = None

def _load(label, path, loader):
    if not os.path.exists(path):
        print(f"  ✗ {label} – not found: {path}")
        return None
    try:
        obj = loader(path)
        print(f"  ✓ {label}")
        return obj
    except Exception as e:
        print(f"  ✗ {label} – {e}")
        return None

print("\nLoading models …")
scaler       = _load("Scaler",            os.path.join(MODEL_DIR, 'scaler.pkl'),            joblib.load)
lr_model     = _load("Linear Regression", os.path.join(MODEL_DIR, 'linear_regression.pkl'), joblib.load)
kmeans_model = _load("K-Means",           os.path.join(MODEL_DIR, 'kmeans_model.pkl'),       joblib.load)

try:
    bp_path = os.path.join(MODEL_DIR, 'backpropagation_model.pkl')
    if os.path.exists(bp_path):
        with open(bp_path, 'rb') as f:
            bp_model = pickle.load(f)
        print("  ✓ Backpropagation")
    else:
        print("  ✗ Backpropagation – not found")
except Exception as e:
    print(f"  ✗ Backpropagation – {e}")

try:
    from tensorflow.keras.models import load_model as keras_load
    ann_model  = _load("ANN",  os.path.join(MODEL_DIR, 'ann_model.h5'),  lambda p: keras_load(p, compile=False))
    lstm_model = _load("LSTM", os.path.join(MODEL_DIR, 'lstm_model.h5'), lambda p: keras_load(p, compile=False))
except Exception as e:
    print(f"  ✗ TensorFlow unavailable – {e}")

metrics_path = os.path.join(MODEL_DIR, 'all_metrics.pkl')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'rb') as f:
            all_metrics = pickle.load(f)
        print("  ✓ Metrics")
    except Exception as e:
        print(f"  ✗ Metrics – {e}")
print()

# ── Dataset summary ───────────────────────────────────────────────────────
_dataset_summary = {}
try:
    df_raw = load_historical_data()
    if df_raw is not None:
        _dataset_summary = {
            'rows':     len(df_raw),
            'start':    str(df_raw['Date'].min()),
            'end':      str(df_raw['Date'].max()),
            'min_rate': round(float(df_raw['Close'].min()), 2),
            'max_rate': round(float(df_raw['Close'].max()), 2),
            'avg_rate': round(float(df_raw['Close'].mean()), 2),
        }
except Exception as e:
    print(f"Dataset summary error: {e}")

# ── Feature computation ────────────────────────────────────────────────────
def compute_features_from_date(year, month, day):
    """
    Menghitung fitur berdasarkan tanggal input.
    Mendukung tanggal historis DAN masa depan.
    Untuk masa depan: menggunakan data 60 hari terakhir dari dataset.
    """
    df = load_historical_data()
    if df is None:
        return None, "Dataset tidak ditemukan."

    try:
        target_date = date(year, month, day)
    except ValueError as e:
        return None, f"Tanggal tidak valid: {e}"

    latest_date = df['Date'].max()
    is_future   = target_date > latest_date

    if is_future:
        # Gunakan 60 hari terakhir dari dataset sebagai basis fitur historis
        df_hist = df.tail(60).copy()
        info_msg = f"future:{target_date}"
    else:
        # Tanggal historis: ambil data hingga tanggal tersebut
        df_hist = df[df['Date'] <= target_date].tail(60).copy()
        info_msg = f"historical:{target_date}"

    if len(df_hist) < 30:
        return None, f"Data historis tidak cukup (minimal 30 hari, tersedia {len(df_hist)} hari)"

    df_hist  = df_hist.reset_index(drop=True)
    prices   = df_hist['Close'].values
    target_dt = datetime(year, month, day)

    features = {
        'day':       target_dt.day,
        'month':     target_dt.month,
        'year':      target_dt.year,
        'dayofweek': target_dt.weekday(),
    }

    features['lag_1']  = float(prices[-1])  if len(prices) >= 1  else 0
    features['lag_3']  = float(prices[-3])  if len(prices) >= 3  else features['lag_1']
    features['lag_7']  = float(prices[-7])  if len(prices) >= 7  else features['lag_1']
    features['lag_14'] = float(prices[-14]) if len(prices) >= 14 else features['lag_1']
    features['lag_30'] = float(prices[-30]) if len(prices) >= 30 else features['lag_1']

    features['rolling_mean_7']  = float(np.mean(prices[-7:]))  if len(prices) >= 7  else features['lag_1']
    features['rolling_mean_14'] = float(np.mean(prices[-14:])) if len(prices) >= 14 else features['lag_1']
    features['rolling_mean_30'] = float(np.mean(prices[-30:])) if len(prices) >= 30 else features['lag_1']
    features['rolling_std_7']   = float(np.std(prices[-7:]))   if len(prices) >= 7  else 0
    features['rolling_std_14']  = float(np.std(prices[-14:]))  if len(prices) >= 14 else 0
    features['rolling_std_30']  = float(np.std(prices[-30:]))  if len(prices) >= 30 else 0

    features['latest_rate'] = float(prices[-1])
    features['is_future']   = is_future
    features['_info']       = info_msg

    return features, None

def _scale(arr):
    if scaler is not None:
        return scaler.transform(arr)
    return arr

def _predict_all(input_array):
    scaled = _scale(input_array)
    if len(scaled.shape) == 1:
        scaled = scaled.reshape(1, -1)
    preds = {}

    if lr_model is not None:
        try:
            preds['linear_regression'] = round(float(lr_model.predict(scaled)[0]), 2)
        except Exception as e:
            preds['linear_regression'] = None
            print(f"LR error: {e}")

    if ann_model is not None:
        try:
            preds['ann'] = round(float(ann_model.predict(scaled, verbose=0)[0][0]), 2)
        except Exception as e:
            preds['ann'] = None
            print(f"ANN error: {e}")

    if lstm_model is not None:
        try:
            lstm_in = np.repeat(scaled.reshape(1, 1, -1), 10, axis=1)
            preds['lstm'] = round(float(lstm_model.predict(lstm_in, verbose=0)[0][0]), 2)
        except Exception as e:
            preds['lstm'] = None
            print(f"LSTM error: {e}")

    if bp_model is not None:
        try:
            preds['backpropagation'] = round(float(bp_model.predict(scaled)[0]), 2)
        except Exception as e:
            preds['backpropagation'] = None
            print(f"BP error: {e}")

    if kmeans_model is not None:
        try:
            km_in = scaled[:, :3] if scaled.shape[1] >= 3 else scaled
            preds['cluster'] = int(kmeans_model.predict(km_in)[0])
        except Exception as e:
            preds['cluster'] = 0
            print(f"K-Means error: {e}")

    return preds

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', dataset=_dataset_summary)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            year  = int(request.form.get('year'))
            month = int(request.form.get('month'))
            day   = int(request.form.get('day'))

            features, error = compute_features_from_date(year, month, day)
            if error:
                return render_template('predict.html', error=error, feature_names=FEATURE_NAMES)

            input_data  = [features[f] for f in FEATURE_NAMES]
            input_array = np.array(input_data).reshape(1, -1)
            predictions = _predict_all(input_array)

            # Format tanggal
            try:
                dt = datetime(year, month, day)
                date_str = dt.strftime('%d %B %Y')
                weekday  = dt.strftime('%A')
            except Exception:
                date_str = f"{day:02d}/{month:02d}/{year}"
                weekday  = ""

            return render_template(
                'predict.html',
                predictions=predictions,
                date_str=date_str,
                weekday=weekday,
                is_future=features.get('is_future', False),
                latest_rate=features.get('latest_rate', 0),
                input_features=features,
                feature_names=FEATURE_NAMES,
                metrics=all_metrics,
                dataset=_dataset_summary,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('predict.html',
                                   error=f"Terjadi kesalahan: {str(e)}",
                                   feature_names=FEATURE_NAMES)

    return render_template('predict.html', feature_names=FEATURE_NAMES, dataset=_dataset_summary)

@app.route('/compare')
def compare():
    return render_template('compare.html', metrics=all_metrics)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if 'year' in data and 'month' in data and 'day' in data:
            features, error = compute_features_from_date(
                int(data['year']), int(data['month']), int(data['day'])
            )
            if error:
                return jsonify({'status': 'error', 'message': error}), 400
            input_data = [features[f] for f in FEATURE_NAMES]
        else:
            input_data = [float(data.get(f, 0)) for f in FEATURE_NAMES]

        input_array = np.array(input_data).reshape(1, -1)
        predictions = _predict_all(input_array)
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'ok',
        'models': {
            'scaler':           scaler is not None,
            'linear_regression': lr_model is not None,
            'ann':              ann_model is not None,
            'lstm':             lstm_model is not None,
            'kmeans':           kmeans_model is not None,
            'backpropagation':  bp_model is not None,
            'metrics':          all_metrics is not None,
        }
    })

@app.route('/api/latest_rate')
def api_latest_rate():
    df = load_historical_data()
    if df is not None:
        latest = df.iloc[-1]
        return jsonify({'status': 'success', 'date': str(latest['Date']), 'rate': float(latest['Close'])})
    return jsonify({'status': 'error', 'message': 'Data not found'}), 404

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
