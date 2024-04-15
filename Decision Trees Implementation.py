from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Dataset
data = {
    'Sector': ['Service Centers', 'Social Joints', 'Health', 'Restaurants'],
    'Number of transactions': [287, 730, 100, 1383]
}

# Convert data to numpy array
X = np.array(data['Number of transactions']).reshape(-1, 1)
y = np.array(data['Sector'])

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Print out the optimal results in a 4 by 4 matrix format
print("Optimal results in decision trees (4 by 4 matrix):")

# Create a matrix to store the predicted sectors
matrix = []

# Predict the sector for each possible combination of transactions
for i in range(len(X)):
    row = []
    for j in range(len(X)):
        prediction = model.predict(np.array([[X[i][0]], [X[j][0]]]))
        row.append(prediction[0])
    matrix.append(row)

# Print the matrix
for row in matrix:
    print("\t".join(row))
