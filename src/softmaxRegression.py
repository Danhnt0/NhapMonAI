import time
from tensorflow.keras.datasets import mnist
import sklearn.preprocessing
import warnings
import pickle
from sklearn.linear_model import LogisticRegression

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0




# flatten the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

time_start = time.time()

# use Softmax to classify 
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter= 1000)

# train the model
model.fit(x_train, y_train)

# test the model
predictions = model.predict(x_test)
score = model.score(x_test, y_test)
print('\n\n\n')
print("The model Softmax accuracy is: {}".format(score))

time_end = time.time()
print("The total time taken is: {} s".format(time_end - time_start))

print('\n\n\n')

# save the model

pickle.dump(model, open('model/softmax.pkl', 'wb'))




# plot the data
# import matplotlib.pyplot as plt
# import numpy as np


# # print total number of correct predictions
# correct = 0
# for i in range(32):
#     if predictions[i] == y_test[i]:
#         correct += 1


# print("The model accuracy is: {}/32".format(correct))


# plt.figure(figsize=(6,6))
# for i in range(32):
#     plt.subplot(8,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_test[i].reshape(28,28), cmap=plt.cm.binary)
#     plt.xlabel("Prediction: {}".format(predictions[i]))

# plt.tight_layout()
# plt.show()