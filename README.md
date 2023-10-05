# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step 1. 
Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.
## Step 2. 
Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.
## Step 3. 
Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.
## Step 4. 
Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.
## Step 5.
Predict the value of y and also print the values of m and c.
## Step 6. 
Plot the accquired graph with respect to hours and scores using the scatter plot function.
## Step 7. 
End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Soundariyan M N
RegisterNumber:  212222230146
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("ex1.txt",header = None)

plt.scatter(data[0], data[1])
plt.xticks(np.arange(5,30, step = 5))
plt.yticks(np.arange(-5,30, step = 5))
plt.xlabel("Population of city (10000)")
plt.ylabel("Profit ($10000)")
plt.title("Profit Prediction")

def computeCost(x, y, theta):
  m = len(y)
  h = x.dot(theta)
  square_arr = (h-y)**2

  return 1/(2*m) * np.sum(square_arr)

data_n = data.values
m = data_n[:,0].size
x = np.append(np.ones((m,1)), data_n[:,0].reshape(m,1),axis = 1)
y = data_n[:,1].reshape(m,1)
theta = np.zeros((2,1))

computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
   m = len(y)
   j_history = []

   for i in range(num_iters):
      prediction = x.dot(theta)
      error = np.dot(x.transpose(), (prediction -y))
      descent = alpha * 1/m * error
      theta -= descent
      j_history.append(computeCost(x,y,theta))

   return theta, j_history

theta, j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("iteration")
plt.ylabel("$(\theta)$")
plt.title("cost function using gradient descent")

plt.scatter(data[0], data[1])
x_value = [x for x in range(25)]
y_value = [y * theta[1] + theta[0] for y in x_value]
plt.plot(x_value, y_value, color = "r")
plt.xticks(np.arange(5,30, step = 5))
plt.yticks(np.arange(-5,30, step = 5))
plt.xlabel("population of the city (10,000)")
plt.ylabel("profit ($10,000)")
plt.title("profit prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population=35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population=35,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180222.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180301.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180334.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180417.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180449.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180522.png)

![model](https://github.com/soundariyan18/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/Screenshot%202023-10-05%20180534.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
