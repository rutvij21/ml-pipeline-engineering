import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Loading dataset
data = load_breast_cancer()

# x is the feature matrix (input data) and y is the target vector (labels)
# x is df and y is series
x =pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

# Splitting data with random_state for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scaling features and training model
# Different features have different scales, and this can affect the performance of some ML algorithms.
# Eg radius = 20, area = 1000, smoothness = 0.1, etc.
# Pipeline allows us to chain multiple steps together, in this case, scaling and modeling,
# so that we can treat them as a single unit. 
# StandardScaler standardizes the features by removing the mean and scaling to unit variance eg -1.2, 0.5,1.8, etc.


# max_iter=10000 is set to ensure that the logistic regression model has enough iterations to converge,
# especially if the dataset is complex or if the default number of iterations (usually 100) is not sufficient 
# for the optimization algorithm to find the optimal parameters.
pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=10000))])

# Fitting the logistic regression model to the training data (x_train and y_train)
# to learn the relationship between the features and the target variable.
pipeline.fit(x_train, y_train)

# Calculating accuracy of the pipeline on the test set
print("Pipeline accuracy:", pipeline.score(x_test, y_test))
# Output : Pipeline accuracy: 0.9736842105263158