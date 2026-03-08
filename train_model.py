from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data=load_breast_cancer()
x=data.data
y=data.target

#Splitting data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#Training model
model = RandomForestClassifier()
model.fit(x_train,y_train)

#Predictions
predictions =model.predict(x_test)

#Evaluation
print(classification_report(y_test,predictions))

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Saving model
pickle.dump(model,open("model.pkl", "wb"))


print("Model saved")

#Inference example
sample=x_test.iloc[0].values.reshape(1,-1)
prediction = model.predict(sample)
