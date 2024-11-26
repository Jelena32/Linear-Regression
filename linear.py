import numpy as np
import matplotlib.pyplot as plt

#Reading training and test data from text files
x_train = np.loadtxt('inputData.txt')
y_train = np.loadtxt('outputData.txt')
x_test = np.loadtxt('inputData_test.txt')
y_test = np.loadtxt('outputData_test.txt')

#Visualition of training data
plt.scatter(x_train, y_train)
plt.xlabel('Ulazne vrednosti (x_train)')
plt.ylabel('Izlazne vrednosti (y_train)')
plt.show()

mse_values=[]
#Finding the best values of b0 and w using gradient descent algorithm
Num = 1000 #Number of epochs
learn_rate = 0.001  #Learning rate
b0=-0.5 #bias
w=-5 #weight
for i in range(Num):
    w = w + learn_rate * np.mean(x_train * (y_train - b0 -w * x_train))
    b0 = b0 + learn_rate * np.mean(y_train - b0 - w * x_train)
    y_predict = w * x_test + b0
    mse_values.append(np.mean((y_predict - y_test) ** 2))

plt.plot(mse_values,Num)
#Prediction of y with best coefficients
y_predict=w*x_test+b0

#MSE of test and predicted values
mse = np.mean((y_predict-y_test)**2)
print(mse)

x_line = np.linspace(min(x_train), max(x_train), 100)
y_line = w * x_line + b0

#Visualition of actual and predicted values with regression line
plt.scatter(x_test, y_test, color='green',label='Test values')
plt.scatter(x_test, y_predict, color='blue',label='Predicted values')
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.xlabel('X feature')
plt.ylabel('Y target')
plt.legend()
plt.show()