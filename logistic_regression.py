from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

#loading and reading the data from csv file using pandas
data = pd.read_csv('exam_score.csv')

#loading the data from csv files and converting into numpy array using .values
X = data[['Exam Score1','Exam Score2']].values
y = data['Pass'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_prediction))
print("\nClassification Report:")
print(classification_report(y_test, y_prediction))

