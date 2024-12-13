#Most of the code has been done by me but I have taken reference from Trensorflow's Documentations & Tutorials. 
# I have provided the citations below extensively. 
#I have also used ChatGPT to refine and structure my code for better perofrmance. 


import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
# Citation:
# Setting a random seed for reproducibility follows best practices outlined in TensorFlow’s documentation: https://www.tensorflow.org/guide/random_numbers.
tf.random.set_seed(1234)

# Load the notMNIST dataset
# Citation:
# The code for loading the dataset using np.load and extracting training and test data is adapted from the notMNIST-Starter provided in CPSC 433 F24 by Jonathan Hudson.
print("--Loading Data--")
with np.load("notMNIST.npz", allow_pickle=True) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

# Normalize the data
# Citation:
# Data normalization and expanding dimensions were adapted from the TensorFlow image data preprocessing tutorial: https://www.tensorflow.org/tutorials/load_data/images.
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Build the improved model
# Citation:
# The model architecture, including additional layers, dropout, and L2 regularization, was based on recommendations from TensorFlow’s overfitting tutorial: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit.
# The use of the Input layer and Sequential API follows the TensorFlow guide for building models: https://www.tensorflow.org/guide/keras/sequential_model.
print("--Building Improved Model--")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Citation:
# The choice of the Adam optimizer and sparse categorical crossentropy loss function is based on TensorFlow’s beginner tutorials for classification tasks: https://www.tensorflow.org/tutorials/keras/classification.
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train the improved model
# Citation:
# The model training process with a validation split was adapted from TensorFlow’s model training and evaluation tutorials: https://www.tensorflow.org/tutorials/keras/classification.
print("--Training Improved Model--")
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2, verbose=2)

# Evaluate the model
# Citation:
# The model evaluation steps using evaluate() and printing formatted results follow best practices shown in TensorFlow’s evaluation and prediction guide: https://www.tensorflow.org/guide/keras/train_and_evaluate.
print("--Evaluating Improved Model--")
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")

# Save the improved model
# Citation:
# The model saving method using model.save() was implemented following the TensorFlow save and load tutorial: https://www.tensorflow.org/tutorials/keras/save_and_load.
print("--Saving Improved Model--")
model.save('notMNIST-Complete.keras')
print("Model saved as notMNIST-Complete.keras")
