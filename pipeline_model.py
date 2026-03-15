import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

x =pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=10000))])

pipeline.fit(x_train, y_train)

print("Pipeline accuracy:", pipeline.score(x_test, y_test))