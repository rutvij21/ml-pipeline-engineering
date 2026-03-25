from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Loading dataset
data=load_breast_cancer()
# Keeping x as numpy array, 
# instead of converting to pandas DataFrame (pd.DataFrame(data.data, columns=data.feature_names))
# This is because RandomForestClassifier or sklearn can work with numpy arrays directly, 
# and it avoids the overhead of creating a DataFrame when it's not necessary for this specific task.
# You can view data if its DataFrame using print(df.head(5)), 
# otherwise in case of numpy array you can use print(data[:5])
# x is the feature matrix (input data) and y is the target vector (labels)
x=data.data
print(data[:5])
y=data.target 

# Splitting data
# We could use random_state = 10 or 0 also, since its just for reproducibility, 
# but 42 is commonly used as a convention in the data science community.
# and this random_state value ensures that the same random split of the data will occur each time you run the code,
#  which is important for consistent results and debugging.
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# Training model
model = RandomForestClassifier()
model.fit(x_train,y_train)

# Predictions
predictions =model.predict(x_test)

# Evaluation
print(classification_report(y_test,predictions))

# confusion matrix
cm = confusion_matrix(y_test,predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Saving model
pickle.dump(model,open("model.pkl", "wb"))


print("Model saved")

# Inference example
# iloc is index-based selection, and we are selecting the first row of x_test, 
# reshaping it to be a 2D array with one sample (row-1D array) and the same number of features as the training data. 
# This is necessary because the predict method expects a 2D array as input.
# x_test[0:5] would give you the first 5 rows of x_test, ie rows 0 to 4
sample = x_test[0].reshape(1, -1)
prediction = model.predict(sample)
print(f"Prediction for the first test sample: {prediction}")

# loc is label-based selection, and since x_test is a numpy array, it does not have labels like a DataFrame would.
# df.loc[:, "mean_radius"] would select all rows (:) and of column/feature name "mean_radius" 

