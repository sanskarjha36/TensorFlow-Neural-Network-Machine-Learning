#Most of the code has been done by me but I have taken reference from Trensorflow's Documentations & Tutorials. 
# I have provided the citations below extensively. 
#I have also used ChatGPT to refine and structure my code for better perofrmance. 


import tensorflow as tf
import pandas as pd
import numpy as np

# Load the training and test datasets
# Citation:
# The code for loading CSV files using pandas follows best practices for data handling as outlined in the pandas documentation: https://pandas.pydata.org/docs/.
print("--Loading Train/Test Data--")
train_data = pd.read_csv('draft_train.csv')
test_data = pd.read_csv('draft_test.csv')

# Replace missing values with 0 (in case any are left)
# Citation:
# The use of fillna() to handle missing values is based on data preprocessing techniques recommended in the pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html.
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Convert 'position' column to numeric using one-hot encoding
# Citation:
# One-hot encoding of the 'position' column using pd.get_dummies() follows standard practices for converting categorical data to numerical features: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html.
train_data = pd.get_dummies(train_data, columns=['position'])
test_data = pd.get_dummies(test_data, columns=['position'])

# Ensure the columns match between train and test data (fix any missing columns)
# Citation:
# The use of align() to match columns between training and testing data is a method recommended by the pandas documentation for data alignment: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.align.html.
train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

# Separate features (X) and labels (y)
# Citation:
# The separation of features and labels using pandas drop() method is adapted from common data preprocessing techniques: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html.
X_train = train_data.drop(columns=['drafted'])
y_train = train_data['drafted']
X_test = test_data.drop(columns=['drafted'])
y_test = test_data['drafted']

# Normalize the numerical features (e.g., height, weight, etc.)
# Citation:
# Normalizing numerical features using mean and standard deviation follows best practices in feature scaling as described in machine learning literature: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling.
numeric_features = ['height', 'weight', 'draftage', 'bench', 'forty', 'fortyelectric', 'verticaljump', 'broadjump', 'threecone', 'shuttle']
X_train[numeric_features] = (X_train[numeric_features] - X_train[numeric_features].mean()) / X_train[numeric_features].std()
X_test[numeric_features] = (X_test[numeric_features] - X_test[numeric_features].mean()) / X_test[numeric_features].std()

# Build the logistic regression model
# Citation:
# The model architecture using Dense layers and a sigmoid activation function for binary classification is based on TensorFlow’s beginner classification guide: https://www.tensorflow.org/tutorials/keras/classification.
# The use of Sequential API and specifying input_shape follows TensorFlow’s model-building best practices: https://www.tensorflow.org/guide/keras/sequential_model.
print("--Building Model--")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
# Citation:
# The choice of the Adam optimizer and binary crossentropy loss function for binary classification tasks is based on TensorFlow’s recommendations: https://www.tensorflow.org/tutorials/keras/classification.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# Citation:
# The training of the model using a validation split of 20% and 50 epochs follows guidelines provided in TensorFlow’s model training and evaluation tutorial: https://www.tensorflow.org/tutorials/keras/classification.
print("--Training Model--")
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

# Evaluate the model on the test data
# Citation:
# The evaluation step using model.evaluate() to compute loss and accuracy is adapted from TensorFlow’s evaluation and prediction guide: https://www.tensorflow.org/guide/keras/train_and_evaluate.
print("--Evaluating Model--")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Save the trained model
# Citation:
# The model saving method using model.save() follows the TensorFlow guide for saving and loading models: https://www.tensorflow.org/tutorials/keras/save_and_load.
model.save('CFLModel.keras')
print("--Model Saved as CFLModel.keras--")
