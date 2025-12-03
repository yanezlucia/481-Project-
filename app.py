from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model (keep this in try block)
try:
    print("Attempting to load model...")
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    train_df = pd.read_csv("train_preprocessed.csv")
    zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                                'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 
                                'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 
                                'PSH Flag Count', 'ECE Flag Count']
    features_to_drop = ['Label', 'Binary_Label'] + zero_importance_features
    X_train = train_df.drop(features_to_drop, axis=1)
    feature_names = X_train.columns.tolist()
    feature_stats = X_train.describe()
    
    print("\n" + "="*60)
    print("Successfully Loaded Model!")
    print("=" * 60)
except Exception as e:
    print(f"Something went wrong: {e}")
    exit(1)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/simulate/<traffic_type>")
def simulate_traffic(traffic_type):
    """Generate Simulated traffic and make predictions"""
    if traffic_type == 'benign':
        simulated_features = generate_benign_traffic()
    elif traffic_type == 'attack':
        simulated_features = generate_attack_traffic()
    else:
        return jsonify({'error': 'Invalid traffic type'}), 400

    features_scaled = scaler.transform([simulated_features])
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    confidence = max(prediction_proba) * 100

    result = {
        'prediction': 'Benign' if prediction == 1 else 'Attack',
        'confidence': round(confidence, 2),
        'features': {feature_names[i]: round(simulated_features[i], 2) 
                    for i in range(min(10, len(feature_names)))}
    }
    return jsonify(result)

# Helper functions (NO indentation)
def generate_benign_traffic():
    features = []
    for col in feature_names:
        mean = feature_stats[col]['mean']
        std = feature_stats[col]['std']
        value = np.random.normal(mean, std * 0.5)
        features.append(max(0, value))
    return features

def generate_attack_traffic():
    features = []
    for col in feature_names:
        mean = feature_stats[col]['mean']
        std = feature_stats[col]['std']
        value = np.random.normal(mean * 1.5, std * 2)
        features.append(max(0, value))
    return features

if __name__ == '__main__':
    app.run(debug=True, port=5000)