from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset1 = loadtxt('Movement.csv', delimiter=',')
dataset2 = loadtxt('NoMovement.csv', delimiter=',')

print(dataset)
