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

    print("\n" + "="*60)

    print("Separating benign and attack samples...")
    benign_samples = train_df[train_df['Binary_Label'] == 1].drop(features_to_drop, axis=1)
    attack_samples = train_df[train_df['Binary_Label'] == 0].drop(features_to_drop, axis=1)

    benign_stats = benign_samples.describe()
    attack_stats = attack_samples.describe()

    print(f"Benign samples: {len(benign_samples)}")
    print(f"Attack samples: {len(attack_samples)}")
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
        print("\n=== BENIGN TRAFFIC GENERATED ===")
    elif traffic_type == 'attack':
        simulated_features = generate_attack_traffic()
        print("\n=== ATTACK TRAFFIC GENERATED ===")
    else:
        return jsonify({'error': 'Invalid traffic type'}), 400
    
    print(f"First 5 features: {simulated_features[:5]}")
    
    features_scaled = scaler.transform([simulated_features])
    print(f"Scaled features (first 5): {features_scaled[0][:5]}")
    
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    print(f"Prediction: {prediction} (0=Attack, 1=Benign)")
    print(f"Probabilities: Attack={prediction_proba[0]:.2f}, Benign={prediction_proba[1]:.2f}")
    
    confidence = max(prediction_proba) * 100

    result = {
        'prediction': 'Benign' if prediction == 1 else 'Attack',
        'confidence': round(confidence, 2),
        'features': {feature_names[i]: round(simulated_features[i], 2) 
                    for i in range(min(10, len(feature_names)))}
    }
    return jsonify(result)

def generate_benign_traffic():
    """Generate realistic benign traffic """
    features = []
    for col in feature_names:
        mean = benign_stats[col]['mean']
        std = benign_stats[col]['std']
        value = np.random.normal(mean, std * 0.3)
        features.append(max(0, value))
    return features

def generate_attack_traffic():
    """Generate realistic attack traffic """
    features = []
    for col in feature_names:
        mean = attack_stats[col]['mean']
        std = attack_stats[col]['std']
        value = np.random.normal(mean, std * 0.5)
        features.append(max(0, value))
    return features

if __name__ == '__main__':
    app.run(debug=True, port=5000)