import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


data = load_breast_cancer()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

log_model = LogisticRegression(max_iter=10000)
log_model.fit(x_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

print("Logistic Regression Accuracy:",  accuracy_score(y_test, log_model.predict(x_test)))


print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(x_test)))

# Logistic Regression Accuracy: 0.9736842105263158
# Random Forest Accuracy: 0.9736842105263158