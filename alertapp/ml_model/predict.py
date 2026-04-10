import pickle
import numpy as np

MODEL_PATH = 'alertapp/ml_model/outbreak_model.pkl'

def predict_outbreak(features):
    """Load the trained model and predict the outcome for given features."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(np.array([features]))
    return prediction[0]