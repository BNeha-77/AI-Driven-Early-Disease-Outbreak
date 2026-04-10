import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('alertapp/data/train.csv')
X = df.drop(['date', 'region', 'outbreak'], axis=1, errors='ignore')
y = df['outbreak']
model = RandomForestClassifier()
model.fit(X, y)
with open('alertapp/ml_model/outbreak_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved to alertapp/ml_model/outbreak_model.pkl")