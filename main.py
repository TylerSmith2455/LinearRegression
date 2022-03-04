from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from linearRegression import LinearRegression
import matplotlib.pyplot as plt

# Normalize wine features
def normalizeWine(data):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data[i][j] = (data[i][j]-min[j])/(max[j]-min[j])

    return data

# Normalize synthetic datasets, create higher order polynomial
def normalizeSynthetic(data, order):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    for i in range(len(data)):
        data[i][1] = (data[i][1]-min[1])/(max[1]-min[1])
    
    for i in range(order-1):
        data = np.insert(data, i+2, 1, axis=1)
        for j in range(len(data)):
            data[j][i+2] = data[j][1]**(i+2)
    
    return data

# Calculate Mean Squared Error
def MSE(regressionModel, data, values):
    return np.mean((regressionModel.predict(data, values))**2)

# Return higher order polynomial for visualization
def polynomial(x, model, order):
    y = model.thetas[0]
    for i in range(order):
        y += model.thetas[i+1] * ((x)**(i+1))
    return y

# Create plots for higher order polynomial regression models
def visualizeModels(data, values, model, order, name):
    xpoints = data[:, 1]
    plt.scatter(xpoints, values)
    x = np.linspace(0,1,10000)
    plt.plot(x, polynomial(x, model, order))
    plt.xlim([0, 1])
    title = name + " " + str(order) + " Order Polynomial"
    plt.title(title)
    plt.xlabel('Normalized Feature Values')
    plt.ylabel('Class Label')
    plt.show()


def main():
    # Read in wine data and normalize it
    wine = pd.read_csv('winequality-red.csv', skiprows=1, header=None)
    wine = normalizeWine(wine.to_numpy().astype(float))
    
    # Seperate features and class label
    wineValues = wine[:, -1]
    wine = np.delete(wine, -1, 1)
    wine = np.insert(wine, 0, 1, axis=1)

    # Create the wine linear regression model
    wineRegression = LinearRegression(.001,10000)
    wineRegression.fit(wine, wineValues, 1.5)

    print(" Wine Dataset")
    print("Mean Squared Error: ", MSE(wineRegression, wine, wineValues))
    print("Weight values: ", *wineRegression.thetas, sep=' ')

    # Read in synthetic1 data and normalize it
    synthetic1 = pd.read_csv('synthetic-1.csv', header = None)
    synthetic1 = synthetic1.to_numpy().astype(float)
    synthetic1_values = synthetic1[:, -1]
    synthetic1 = np.delete(synthetic1, -1, 1)
    synthetic1 = np.insert(synthetic1, 0, 1, axis=1)
    print('\n', "Synthetic-1 Dataset")

    # Synthetic 1 2nd order ploynomial
    synthetic1_2 = normalizeSynthetic(synthetic1, 2)
    #print(synthetic1_2)
    regression = LinearRegression(.01,100000)
    regression.fit(synthetic1_2, synthetic1_values, 35)
    print("2nd Order Polynomial MSE: ", MSE(regression, synthetic1_2, synthetic1_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic1_2, synthetic1_values, regression, 2, "Synthetic-1")

    # Synthetic 1 3rd order polynomial
    synthetic1_3 = normalizeSynthetic(synthetic1, 3)
    regression = LinearRegression(1,100000)
    regression.fit(synthetic1_3, synthetic1_values, 10)
    print("3rd Order Polynomial MSE: ", MSE(regression, synthetic1_3, synthetic1_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic1_3, synthetic1_values, regression, 3, "Synthetic-1")

    # Synthetic 1 5th order polynomial
    synthetic1_5 = normalizeSynthetic(synthetic1, 5)
    regression = LinearRegression(1,100000)
    regression.fit(synthetic1_5, synthetic1_values, 10)
    print("5th Order Polynomial MSE: ", MSE(regression, synthetic1_5, synthetic1_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic1_5, synthetic1_values, regression, 5, "Synthetic-1")

    # Read in synthetic2 data and normalize it
    synthetic2 = pd.read_csv('synthetic-2.csv', header = None)
    synthetic2 = synthetic2.to_numpy().astype(float)
    synthetic2_values = synthetic2[:, -1]
    synthetic2 = np.delete(synthetic2, -1, 1)
    synthetic2 = np.insert(synthetic2, 0, 1, axis=1)
    print('\n', "Synthetic-2 Dataset")

    # Synthetic 2 2nd order ploynomial
    synthetic2_2 = normalizeSynthetic(synthetic2, 2)
    regression = LinearRegression(.01,100000)
    regression.fit(synthetic2_2, synthetic2_values, .5)
    print("2nd Order Polynomial MSE: ", MSE(regression, synthetic2_2, synthetic2_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic2_2, synthetic2_values, regression, 2, "Synthetic-2")

    # Synthetic 2 3rd order polynomial
    synthetic2_3 = normalizeSynthetic(synthetic2, 3)
    regression = LinearRegression(.01,100000)
    regression.fit(synthetic2_3, synthetic2_values, .5)
    print("3rd Order Polynomial MSE: ", MSE(regression, synthetic2_3, synthetic2_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic2_3, synthetic2_values, regression, 3, "Synthetic-2")

    # Synthetic 2 5th order polynomial
    synthetic2_5 = normalizeSynthetic(synthetic2, 5)
    regression = LinearRegression(.01,100000)
    regression.fit(synthetic2_5, synthetic2_values, .5)
    print("5th Order Polynomial MSE: ", MSE(regression, synthetic2_5, synthetic2_values))
    print("Weight values: ", *regression.thetas, sep=' ')
    #visualizeModels(synthetic2_5, synthetic2_values, regression, 5, "Synthetic-2")


if __name__ == "__main__":
    main()
    
