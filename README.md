# 481-Project-
DDOS detector with AI integration 

### Load virtual environment 
`venv/scripts/activate` | <= Windows


`source venv/bin/activate` | <= Mac

### Install Req's
`pip install -r requirements.txt`

## What was done
* Separated X features from target feature y. 
* Scaled Features
* Created random_forest model. 

## MODEL PERFORMANCE
```
F1 Score: 0.8131

Classification Report:
              precision    recall  f1-score   support

      Attack       0.98      0.75      0.85     78743
      Benign       0.70      0.97      0.81     46427

    accuracy                           0.83    125170
   macro avg       0.84      0.86      0.83    125170
weighted avg       0.88      0.83      0.84    125170
```