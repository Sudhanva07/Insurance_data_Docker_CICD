import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle


data = pd.read_csv('insurance.csv')
encoded_data = data.copy()

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the categorical variables
encoded_data['sex'] = label_encoder.fit_transform(encoded_data['sex'])
encoded_data['smoker'] = label_encoder.fit_transform(encoded_data['smoker'])
encoded_data['region'] = label_encoder.fit_transform(encoded_data['region'])

encoded_data['charges'] = np.log(encoded_data['charges'])

# Split the data into training and testing sets
X = encoded_data.drop('charges', axis=1)  # Features (all columns except 'charges')
y = encoded_data['charges']  # Target variable ('charges')


model = LinearRegression()
polynomial_converter = PolynomialFeatures(degree=2, include_bias = False)
poly_features = polynomial_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)




# Assume model is your trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)