#Most of the code has been done by me but I have taken reference from Trensorflow's Documentations & Tutorials. 
# I have provided the citations below extensively. 
#I have also used ChatGPT to refine and structure my code for better perofrmance. 


import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
# Citation:
# Setting a random seed ensures reproducible results, which is a common best practice recommended in the TensorFlow guide on random numbers: https://www.tensorflow.org/guide/random_numbers.
tf.random.set_seed(1234)

# Load the notMNIST dataset
# Citation:
# The code to load the notMNIST dataset using np.load and extract training and test data is adapted from the notMNIST-Starter code provided in CPSC 433 F24 by Jonathan Hudson.
print("--Loading Data--")
with np.load("notMNIST.npz", allow_pickle=True) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

# Normalize the data
# Citation:
# Data normalization (scaling pixel values to [0, 1]) and expanding dimensions to match the input shape were adapted from TensorFlow’s image preprocessing tutorials: https://www.tensorflow.org/tutorials/load_data/images.
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define the improved model
# Citation:
# The architecture of the improved model, including additional Dense layers, Batch Normalization, Dropout regularization, and L2 regularization, was based on recommendations from TensorFlow’s tutorial on overfitting: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit.
# The use of the Input layer and Sequential API follows best practices outlined in TensorFlow’s model building guide: https://www.tensorflow.org/guide/keras/sequential_model.
print("--Building Improved Model--")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the improved model
# Citation:
# The choice of the Adam optimizer with a reduced learning rate and the sparse categorical crossentropy loss function is based on TensorFlow’s official guide for classification problems: https://www.tensorflow.org/tutorials/keras/classification.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Implement early stopping
# Citation:
# Early stopping was implemented to prevent overfitting by monitoring the validation loss, following TensorFlow’s callback tutorial: https://www.tensorflow.org/guide/keras/early_stopping.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the improved model
# Citation:
# The training process with a validation split and the use of early stopping were inspired by TensorFlow’s classification and overfitting tutorials: https://www.tensorflow.org/tutorials/keras/classification.
print("--Training Improved Model--")
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=2)

# Evaluate the model
# Citation:
# The evaluation of the model using model.evaluate() for both training and test data follows standard practice as shown in TensorFlow’s evaluation guide: https://www.tensorflow.org/guide/keras/train_and_evaluate.
print("--Evaluating Improved Model--")
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")

# Save the improved model
# Citation:
# Saving the trained model using model.save() was based on the TensorFlow tutorial for saving and loading models: https://www.tensorflow.org/tutorials/keras/save_and_load.
print("--Saving Improved Model--")
model.save('notMNIST-Complete.keras')
print("Model saved as notMNIST-Complete.keras")
