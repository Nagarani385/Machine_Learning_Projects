from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

#loading and reading the data from csv file using pandas
data = pd.read_csv('updated_pollution_dataset.csv')

#loading the data from csv files and converting into numpy array using .values
X = data[['Temperature','Humidity','PM2.5','PM10','NO2','SO2','CO','Population_Density']].values
y = data['Air Quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter= 5000)
# Train the model
model.fit(X_train, y_train)
train_accuracy = accuracy_score(y_train, y_train)
y_prediction = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_prediction))
print("\nClassification Report:")
print(classification_report(y_test, y_prediction))
