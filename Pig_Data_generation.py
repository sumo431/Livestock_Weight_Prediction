import pandas as pd
import numpy as np

n_samples = 200
data = {
    'animal_type' : np.random.choice(['Horse', 'Cow', 'Sheep', 'Goat'], n_samples),
    'age' : np.random.randint(1, 10, n_samples),
    'sex' : np.random.choice(['Male', 'Female'], n_samples),
    'variety' : np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
    'weight': np.random.uniform(50.0, 800.0, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('livestock_data.csv', index=False)
print("Data generated and saved to 'livestock_data.csv.")

