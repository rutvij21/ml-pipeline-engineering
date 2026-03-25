import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Loading dataset
data = load_breast_cancer()

# Converting to pandas DataFrame
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Training Logistic Regression model
# max_iter=10000 is set to ensure that the logistic regression model has enough iterations to converge,
# especially if the dataset is complex or if the default number of iterations (usually 100) is not sufficient 
# for the optimization algorithm to find the optimal parameters.
log_model = LogisticRegression(max_iter=10000)
# Fitting the logistic regression model to the training data (x_train and y_train)
#  to learn the relationship between the features and the target variable.
log_model.fit(x_train, y_train)


# Training Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Calculating accuracy of the logistic regression model on the test set
print("Logistic Regression Accuracy:",  accuracy_score(y_test, log_model.predict(x_test)))

# Calculating accuracy of the random forest model on the test set
print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(x_test)))

# Outputs:
# Logistic Regression Accuracy: 0.9824561403508771
# Random Forest Accuracy: 0.9385964912280702