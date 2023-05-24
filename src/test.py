import time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


# load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_test = x_test / 255.0

#load the model
model = tf.keras.models.load_model('model/model.h5')
C_model = tf.keras.models.load_model('model/C_model.h5')
knn = pickle.load(open('model/knn.pkl', 'rb'))
softmax = pickle.load(open('model/softmax.pkl', 'rb'))

time_start_nn = time.time()
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
time_end_nn = time.time()


time_start_C = time.time()
C_predictions = C_model.predict(x_test)
C_predictions = np.argmax(C_predictions, axis=1)
time_end_C = time.time()

time_start_knn = time.time()
knn_predictions = knn.predict(x_test.reshape(10000, 784))
time_end_knn = time.time()

time_start_softmax = time.time()
softmax_predictions = softmax.predict(x_test.reshape(10000, 784))
time_end_softmax = time.time()


# find the recall, precision, f1_score, accuracy for each model
knn_recall = recall_score(y_test, knn_predictions, average='macro')
knn_precision = precision_score(y_test, knn_predictions, average='macro')
knn_f1_score = f1_score(y_test, knn_predictions, average='macro')
knn_accuracy = accuracy_score(y_test, knn_predictions)

softmax_recall = recall_score(y_test, softmax_predictions, average='macro')
softmax_precision = precision_score(y_test, softmax_predictions, average='macro')
softmax_f1_score = f1_score(y_test, softmax_predictions, average='macro')
softmax_accuracy = accuracy_score(y_test, softmax_predictions)

C_recall = recall_score(y_test, C_predictions, average='macro')
C_precision = precision_score(y_test, C_predictions, average='macro')
C_f1_score = f1_score(y_test, C_predictions, average='macro')
C_accuracy = accuracy_score(y_test, C_predictions)

NN_recall = recall_score(y_test, predictions, average='macro')
NN_precision = precision_score(y_test, predictions, average='macro')
NN_f1_score = f1_score(y_test, predictions, average='macro')
NN_accuracy = accuracy_score(y_test, predictions)

# print the results

import pandas as pd
# create a dataframe to store the results containing recall, precision, f1_score, accuracy and time prediction
print('\n\n')
df = pd.DataFrame({'Model': ['KNN', 'Softmax', 'CNN', 'Dense'],
                     'Recall': [knn_recall, softmax_recall, C_recall, NN_recall],
                        'Precision': [knn_precision, softmax_precision, C_precision, NN_precision],
                        'F1_score': [knn_f1_score, softmax_f1_score, C_f1_score, NN_f1_score],
                        'Accuracy': [knn_accuracy, softmax_accuracy, C_accuracy, NN_accuracy],
                        'Time(s)': [time_end_knn - time_start_knn, time_end_softmax - time_start_softmax, time_end_C - time_start_C, time_end_nn - time_start_nn]})
print(df)

print('\n\n')

# confusion matrix of each model 
# cm = confusion_matrix(y_test, knn_predictions)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title('Confusion matrix of KNN')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# cm = confusion_matrix(y_test, softmax_predictions)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title('Confusion matrix of Softmax')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# cm = confusion_matrix(y_test, C_predictions)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title('Confusion matrix of CNN')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# cm = confusion_matrix(y_test, predictions)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title('Confusion matrix of NN')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# classification report of each model
print('\n\n')
print('Classification report of KNN')
print(classification_report(y_test, knn_predictions))

print('\n\n')
print('Classification report of Softmax')
print(classification_report(y_test, softmax_predictions))

print('\n\n')
print('Classification report of CNN')
print(classification_report(y_test, C_predictions))

print('\n\n')
print('Classification report of NN')
print(classification_report(y_test, predictions))