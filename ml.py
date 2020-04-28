from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


dataset = loadtxt('data.csv', delimiter=',')


X = dataset[:,0:10]
y = dataset[:,10]

model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(len(X)):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
