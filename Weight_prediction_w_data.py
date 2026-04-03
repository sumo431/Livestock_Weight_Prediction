import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('')

X = df[['animal_type', 'age', 'gender', 'variety']]
