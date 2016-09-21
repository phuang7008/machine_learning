#!/usr/bin/python

from statistics import mean
import numpy as np
import random

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# here we are going to generate a random dataset for future usage
def create_dataset(num, variance, step=2, correlation=False):
    val = 1
    y = []
    for i in range(num):
        yy = val + random.randrange(-variance, variance)
        y.append(yy)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    x = [i for i in range(len(y))]
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)

# calculate the line parameters
def slope_and_intercept(x, y):
    m = ( (mean(x) * mean(y) - mean(x*y) ) / 
          (mean(x) * mean(x) - mean(x*x))  )
          
    b = mean(y) - m * mean(x)  
    return m, b

# get some random data
xs, ys = create_dataset(40, 20, 2, 'pos')
print("x values are");  print(xs)
print("y values are");  print(ys)
m, b = slope_and_intercept(xs, ys)
print(m, b)
regression_line = [ m*x + b for x in xs]

# now need to calculate the R-square (coefficient of determination)
def square_error(y_orig, y_line):
    return sum((y_orig - y_line)**2)
    
def coefficient_of_determination(y_orig, y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    square_error_regression = square_error(y_orig, y_line)
    square_error_y_mean = square_error(y_orig, y_mean_line)
    return 1 - (square_error_regression / square_error_y_mean)
    
cod = coefficient_of_determination(ys, regression_line)
print("COD is %f" % cod)
    
    
    