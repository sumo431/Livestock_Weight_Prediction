import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


# Load the dataset
df = pd.read_csv('pig_data.csv')

#devide X and Y as features and target variable
X = df[['animal_type', 'age_months', 'sex', 'variety']]
Y = df['weight_kg']

# Split the dataset into training and testing sets
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), ['animal_type', 'sex', 'variety'])
], remainder='passthrough')

X = ct.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

#Predict only the first 10 items
X_test_sample = X_test[:10]
Y_test_sample = Y_test[:10]
predictions_sample = model.predict(X_test_sample)

results = pd.DataFrame({
    'Type': df.loc[Y_test_sample.index, 'animal_type'].values,
    'Sex': df.loc[Y_test_sample.index, 'sex'].values,
    'Variety': df.loc[Y_test_sample.index, 'variety'].values,
    'Actual': Y_test_sample.values,
    'Predicted': predictions_sample.round(2) 
})

print("\nComparison of Prediction Results")
print(results)