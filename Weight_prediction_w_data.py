from statistics import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


# Load the dataset
df = pd.read_csv('')

#devide X and Y as features and target variable
X = df[['animal_type', 'age', 'sex', 'variety']]
Y = df['weight']

# Split the dataset into training and testing sets
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), ['animal_type', 'sex', 'variety'])
], remainder='passthrough')

X = ct.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print(predictions)


