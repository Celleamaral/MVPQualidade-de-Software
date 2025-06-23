import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00485/tripadvisor_review.csv'
df = pd.read_csv(url)

X = df.drop(columns=['User_Id', 'Review_Date', 'Review_Text', 'Overall_Rating'], errors='ignore')
y = df['Overall_Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
joblib.dump(pipe, 'modelo_review.pkl')
