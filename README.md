# AI-DDoS Detector
An intelligent DDoS attack detection system using machine learning to classify network traffic as benign or malicious. This project implements a Random Forest classifier with feature selection optimization and includes a Flask-based web interface for real-time predictions.

The project is available here: https://ai-ddos-detector.onrender.com

Otherwise you can follow the steps below to get the model working, but that requires the model to be trained. 

### Create venv
`python venv -m .venv`

### Activate Venv
`venv/scripts/activate` | <= Windows
`source venv/bin/activate` | <= Mac

### Install Req's
`pip install -r requirements.txt`

### Workflow
```
# 1. Preprocess the data
python preprocess.py

# 2. Train the model
python train.py

# 3. Launch the Flask application
python app.py
```


## Datasets
Can be found here: https://huggingface.co/datasets/HallowsYves/CPSC481-data
