import time
from tensorflow.keras.datasets import mnist
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# use KNN to classify the data

model = KNeighborsClassifier(n_neighbors= 19)

# flatten the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

time_start = time.time()

# train the model
model.fit(x_train, y_train)

# test the model
score = model.score(x_test, y_test)
print('\n\n\n')
print("The model KNN accuracy is: {}".format(score))

time_end = time.time()
print("The total time taken is: {} s".format(time_end - time_start))
print('\n\n\n')
# save the model

pickle.dump(model, open('model/knn.pkl', 'wb'))

