#This is a practice run—inserting 
# five data points before loading 
# the actual data—to verify that everything is functioning correctly.
import numpy as np
from sklearn.linear_model import LinearRegression

# age, gender, variety
x_train = np.array([
    [2, 0, 0],
    [3, 1, 0],
    [1, 0, 1],
    [4, 1, 1],
    [5, 0, 0]
])

#weight
y_train = np.array([100, 150, 80, 200, 250])

model = LinearRegression()
model.fit(x_train, y_train)

new_animal_1 = np.array([[3, 1, 0]])
new_animal_2 = np.array([[4, 0, 1]])
new_animal_3 = np.array([[1, 0, 0]])


predicted_weight_1 = model.predict(new_animal_1)
predicted_weight_2 = model.predict(new_animal_2)
predicted_weight_3 = model.predict(new_animal_3)


print(f"Predicted weight for the new animal: {predicted_weight_1[0]:.2f} kg")
print(f"Predicted weight for the new animal: {predicted_weight_2[0]:.2f} kg")
print(f"Predicted weight for the new animal: {predicted_weight_3[0]:.2f} kg")

