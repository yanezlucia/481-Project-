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

    zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                                'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 
                                'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 
                                'PSH Flag Count', 'ECE Flag Count']
    
    print("\n" + "="*60)
    print("Loading pre-calculated statistics...")
    
    # Load pre-calculated statistics
    benign_stats = pd.read_json('benign_stats.json')
    attack_stats = pd.read_json('attack_stats.json')
    
    # Get feature names from the statistics
    feature_names = benign_stats.columns.tolist()
    
    print("Successfully Loaded Model and Statistics!")
    print(f"Number of features: {len(feature_names)}")
    print("=" * 60)

    
except Exception as e:
    print(f"Something went wrong: {e}")
    exit(1)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/simulate/real/<traffic_type>")
def simulate_real_traffic(traffic_type):
    """
        Get random samples from dataset
        1. Load the entire dataset
        2. Filter based on passed traffic type 
        3. Pick a random row from the filtered traffic.
    """
    test_df_full = pd.read_csv("sampled_test_data.csv")

    if traffic_type == 'benign':
        benign_samples = test_df_full[test_df_full['Binary_Label'] == 1]
        if len(benign_samples) == 0:
            return jsonify({'error': "No benign samples found."}), 400
        sample = benign_samples.sample(n=1).iloc[0]
        actual_label = "Benign"
    
    elif traffic_type == 'attack':
        attack_samples = test_df_full[test_df_full['Binary_Label'] == 0]
        if len(attack_samples) == 0:
            return jsonify({'error': "No attack samples found."}), 400
        sample = attack_samples.sample(n=1).iloc[0]
        actual_label = "Attack"
    else:
        return jsonify({'error': 'Invalid traffic type'}), 400
    
    # Extract features (drop labels) and convert to array
    sample_features = sample.drop(['Label', 'Binary_Label'] + zero_importance_features).values

    features_scaled = scaler.transform([sample_features])

    # make predictions
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    confidence = max(prediction_proba) * 100

    predicted_label = 'Benign' if prediction == 1 else 'Attack'
    is_correct = (predicted_label == actual_label)
    
    print(f"Actual: {actual_label}, Predicted: {predicted_label}, Correct: {is_correct}")
    print(f"Confidence: {confidence:.2f}%")
    
    result = {
        'prediction': predicted_label,
        'confidence': round(confidence, 2),
        'actual': actual_label,
        'correct': is_correct,
        'features': {feature_names[i]: round(float(sample_features[i]), 2) 
                    for i in range(min(10, len(feature_names)))}
    }
    
    return jsonify(result)


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
        'features': {feature_names[i]: round(float(simulated_features[i]), 2) 
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