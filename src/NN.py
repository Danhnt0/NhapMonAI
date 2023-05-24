import time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset = tfds.load('mnist')

train_dataset = dataset['train']
test_dataset = dataset['test']


# Preprocess the data
def preprocess(data):
    image = data['image']
    label = data['label']
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(32)
test_dataset = test_dataset.map(preprocess).batch(32)

# Build the model

# dense modelz
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# convolutional model
C_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

C_model.summary()

# class Callback to check the accuracy of the training

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True



# Compile the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

C_model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the model

# train the model without callbacks to see the difference of the accuracy
time_start = time.time()

# history = model.fit(train_dataset, epochs=20,batch_size=32, callbacks=[myCallback()])
history = model.fit(train_dataset, epochs=15,batch_size=32)
time_end = time.time()

print('\n\n\n')
print("The total time NN taken is: {} s".format(time_end - time_start))


test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy NN model:', test_acc)


time_start_ = time.time()
C_history = C_model.fit(train_dataset, epochs=15,batch_size=32)

# C_history = C_model.fit(train_dataset, epochs=20,batch_size=32, callbacks=[myCallback()])
time_end_ = time.time()

# print('\n\n\n')
# print("The total time CNN taken is: {} s".format(time_end - time_start))
# print('\n')

# test_loss, test_acc = C_model.evaluate(test_dataset)
# print('Test accuracy CNN_model:', test_acc)
# print('Epoch number:', len(C_history.history['accuracy']))

# train the model with callbacks to see the difference of the epochs

# Evaluate the model
# test_loss, test_acc = model.evaluate(test_dataset)
# print('Test accuracy model:', test_acc)
print("The total time NN taken is: {} s".format(time_end - time_start))
print('Epoch number:', len(history.history['accuracy']))
print('\n')
# test_loss, test_acc = C_model.evaluate(test_dataset)
# print('Test accuracy C_model:', test_acc)
print("The total time CNN taken is: {} s".format(time_end_ - time_start_))
print('Epoch number:', len(C_history.history['accuracy']))

# Plot the accuracy  of the model and the C_model in the same graph


plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(C_history.history['accuracy'], label = 'CNN_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# plot loss
plt.show()

plt.plot(history.history['loss'], label = 'loss')
plt.plot(C_history.history['loss'], label = 'CNN_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')

plt.show()

model.save('model/model.h5')
C_model.save('model/C_model.h5')

