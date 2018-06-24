import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from adalinegd import AdalineGD
from adalinesgd import AdalineSGD
import pdr


# get the iris data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
	header = None)

# Plot 100 samples of the data
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()

# Standardize the data
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Create the AdalineGD model
model1 = AdalineGD(n_iter = 15, eta = 0.01)

# Train the model
model1.fit(X_std, y)

# Plot the training error
plt.plot(range(1, len(model1.cost_) + 1), model1.cost_, marker = 'o', color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

# Plot the decision boundary
pdr.plot_decision_regions(X_std, y, classifier = model1)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()

# Create the AdalineSGD model
model2 = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)

# Train the model
model2.fit(X_std, y)

# Plot the training errors of both of the models
plt.plot(range(1, len(model2.cost_) + 1), model2.cost_, marker = 'x', color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()


# Plot the decision boundary
pdr.plot_decision_regions(X_std, y, classifier = model2)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()
