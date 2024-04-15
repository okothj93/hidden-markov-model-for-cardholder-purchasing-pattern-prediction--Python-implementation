from sklearn.naive_bayes import GaussianNB
import numpy as np

# Dataset
data = {
    'Sector': ['Service Centers', 'Social Joints', 'Health', 'Restaurants'],
    'Number of transactions': [287, 730, 100, 1383]
}

# Convert data to numpy array
X = np.array(data['Number of transactions']).reshape(-1, 1)
y = np.array(data['Sector'])

# Binning the transaction numbers into categories
# You can adjust the bin edges as per your requirement
bins = [0, 300, 600, 900, np.inf]
labels = ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4']

# Assigning each transaction to a bin
X_binned = np.digitize(X, bins=bins, right=True)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_binned, y)

# Print out the optimal results in a 4 by 4 matrix format
print("Optimal results in Naive Bayes (4 by 4 matrix):")

# Create a matrix to store the predicted sectors
matrix = []

# Predict the sector for each possible combination of transactions
for i in range(len(X_binned)):
    row = []
    for j in range(len(X_binned)):
        prediction = model.predict(np.array([[X_binned[i][0]], [X_binned[j][0]]]))
        row.append(prediction[0])
    matrix.append(row)

# Print the matrix
for row in matrix:
    print("\t".join(row))
