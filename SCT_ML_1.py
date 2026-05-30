#Wap to predict the prices of houses based on their no.of bedrooms and bathrooms.

import pandas as pd #for data handling and manipulation
import numpy as np #for stats calculation
from sklearn.model_selection import train_test_split #split into training data and testing data
from sklearn.linear_model import LinearRegression #for input and output variable
from sklearn.metrics import mean_squared_error #for prediction accuracy

data = pd.read_csv("house_price.csv") #read csv file and convert it into table format

print("Dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

X = data[['area', 'bedrooms', 'bathrooms']] #input variable
y = data['price'] #output variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) #20% data for testing ,42 states for constantly checking

model = LinearRegression() #creates model.

model.fit(X_train, y_train) #train the model

# Predict prices
predictions = model.predict(X_test)

# Check error
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# Predict new house price
new_house = pd.DataFrame({
    'area': [2500],
    'bedrooms': [3],
    'bathrooms': [2]
})

price = model.predict(new_house)


print("Predicted Price:", price)

print("\nPredicted Price for House:")
print(price)