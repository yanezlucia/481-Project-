from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    print("Attempting to load model...")
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    train_df = pd.read_csv("train_preprocessed.csv")
    zero_importance_features = ['FIN Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', 'PSH Flag Count', 'ECE Flag Count']
    features_to_drop = ['Label', 'Binary_Label'] + zero_importance_features
    X_train = train_df.drop(features_to_drop, axis=1)
    feature_names = X_train.columns.tolist()

    feature_stats = X_train.describe()
    print("\n" + "="*60)
    print("Successfully Loaded Model!")
    print("=" * 60)

    @app.route("/")
    def home():
        return render_template('index.html')
    
    @app.route("simulate/<traffic_type>")
    def simulate_traffic(traffic_type):
        """Generate Simulated traffic and make predictions"""
        if traffic_type == 'benign':
            simulated_features = generate_benign_traffic()
        elif traffic_type == 'attack':
            simulated_features = generate_attack_traffic()
        else:
            return jsonify({'error': 'Invalid traffic type'}), 400

    def generate_benign_traffic():
        pass

    def generate_attack_traffic():
        pass


except Exception as e:
    print(f"Something when wrong: {Exception}")

