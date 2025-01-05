from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('C:/My_work/ML_projects/kaggle/Student_marks/Student_Marks.csv')
X = data[['number_courses','time_study']].values
y = data['Marks'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_prediction = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_prediction)
r2 = r2_score(Y_test, Y_prediction)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
