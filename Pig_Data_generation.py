import pandas as pd
import numpy as np

n_samples = 200
age_months = np.random.uniform(0, 7, n_samples)


# Birth (0 months): Approx. 1.4 kg (3 lbs)
# Weaning (approx. 3 weeks): Approx. 5–7 kg (12–15 lbs)
# Market (approx. 6 months): Approx. 127 kg (280 lbs)
Weight = 1.4 + (age_months * 21) + np.random.normal(0, 5, n_samples)

data = {
    'animal_type' : np.full(n_samples, 'Pig'),
    'age_months' : np.round(age_months, 1),
    'sex': np.random.choice(['Male', 'Female'], n_samples),
    'variety': np.random.choice(['Berkshire', 'Hampshire', 'Duroc'], n_samples),
    'weight_kg': np.round(Weight, 1)
    #'sex' : np.random.choice(['Male', 'Female'], n_samples),
    #'variety' : np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
    #'weight': np.random.uniform(50.0, 800.0, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('pig_data.csv', index=False)
print("Data generated and saved to 'pig_data.csv'.")
