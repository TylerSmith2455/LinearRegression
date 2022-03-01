import numpy as np

# Linear Regression Class
class LinearRegression:
    def __init__(self, alpha, steps) -> None:
        self.alpha = alpha
        self.steps = steps
        self.thetas = None

    # Create linear regression model
    def fit(self, data, values, targetLoss):
        self.thetas = np.random.uniform(-1,1,len(data[0]))
        
        for i in range(self.steps):
            # Calulate gradient and update theta values
            gradients = (1/len(data)) * np.dot(data.transpose(), self.predict(data, values))
            self.thetas -= self.alpha*gradients
            
            # Calculate loss, If loss is low enough stop
            loss = np.mean((self.predict(data, values))**2)/2
            if loss < targetLoss/2:
                break
    
    # Predict method
    def predict(self, data, values):
        predictions = np.dot(data, self.thetas)
        return predictions - values





        

    