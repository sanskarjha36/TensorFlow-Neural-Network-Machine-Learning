#Most of the code has been done by me but I have taken reference from Trensorflow's Documentations & Tutorials. 
# I have provided the citations below extensively. 
#I have also used ChatGPT to refine and structure my code for better perofrmance. 


# Original Author: Jonathan Hudson
# CPSC 433 F24

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
     class_names, data = check_args()
     x_test, y_test = data
     print(f"--Load Model {sys.argv[2]}--")

# Load the model specified in the command-line argument (e.g., "notMNIST-Partial.keras")
# Citation:
# This line of code uses tf.keras.models.load_model() to load a pre-trained model file, following the TensorFlow save and load tutorial: https://www.tensorflow.org/tutorials/keras/save_and_load.
     model = tf.keras.models.load_model(sys.argv[2])
# Citation:
# The use of model.predict() to generate predictions on the entire test dataset was inspired by TensorFlow’s image classification tutorial: https://www.tensorflow.org/tutorials/keras/classification.
     pick = input(f"Pick test_image (0 -> {len(x_test) - 1}): ")
     predictions = model.predict(x_test)

# Loop to pick test images and display predictions
     while pick.isdigit() and 0 <= int(pick) < len(x_test):
          pick = int(pick)
          img = x_test[pick]
          true_label = y_test[pick]
          print(f"--Should be Class {true_label}--")
          predict(model, class_names, img, true_label)
          pick = input(f"Pick test_image (0 -> {len(x_test) - 1}): ")

     print("Done")

# Citation:
# Expanding the image dimensions using np.expand_dims() was adapted from TensorFlow’s guide on making predictions with a trained model: https://www.tensorflow.org/tutorials/keras/classification.
def predict(model, class_names, img, true_label):
     img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input

# Citation:
# The use of model.predict() to get the prediction array for a single image follows the TensorFlow classification tutorial: https://www.tensorflow.org/tutorials/keras/classification.
     prediction = model.predict(img)[0]  # Get the prediction array for the image

# Determine the predicted label
# Citation:
# The use of np.argmax() to determine the predicted label from the output probabilities is based on TensorFlow’s classification example: https://www.tensorflow.org/tutorials/keras/classification.
     predicted_label = np.argmax(prediction)

     plot(class_names, prediction, true_label, predicted_label, img[0])
     plt.show()

def check_args():
     if len(sys.argv) == 1:
          print("No arguments so using defaults")
          if input("Y for MNIST, otherwise notMNIST:") == "Y":
               sys.argv = ["predict_test.py", "MNIST", "MNIST.keras"]
          else:
               sys.argv = ["predict_test.py", "notMNIST", "notMNIST-Partial.keras"]

     if len(sys.argv) != 3:
          print("Usage: python predict_test.py <MNIST,notMNIST> <model.keras>")
          sys.exit(1)

     if sys.argv[1] == "MNIST":
          print("--Dataset MNIST--")
          class_names = list(range(10))
          mnist = tf.keras.datasets.mnist
          (x_train, y_train), (x_test, y_test) = mnist.load_data()
          x_train, x_test = x_train / 255.0, x_test / 255.0
          data = (x_test, y_test)

     elif sys.argv[1] == "notMNIST":
          print("--Dataset notMNIST--")
          class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
          with np.load("notMNIST.npz", allow_pickle=True) as f:
               x_train, y_train = f['x_train'], f['y_train']
               x_test, y_test = f['x_test'], f['y_test']
          x_train, x_test = x_train / 255.0, x_test / 255.0
          data = (x_test, y_test)

     else:
          print(f"Choose MNIST or notMNIST, not {sys.argv[1]}")
          sys.exit(2)
# Citation:
# The argument check for the model file extension follows best practices for input validation in Python scripts, as described in Python’s official documentation: https://docs.python.org/3/tutorial/errors.html.
     if not sys.argv[2].endswith(".keras"):
          print(f"{sys.argv[2]} is not a valid keras model file")
          sys.exit(3)

     return class_names, data

def plot(class_names, prediction, true_label, predicted_label, img):
     plt.figure(figsize=(6, 3))
     plt.subplot(1, 2, 1)
     plt.grid(False)
     plt.xticks([])
     plt.yticks([])
     plt.imshow(img, cmap=plt.cm.binary)

# Display the predicted label with confidence percentage
     if predicted_label == true_label:
          color = 'blue'
     else:
          color = 'red'
# Citation:
# Displaying the predicted class label with its confidence percentage, and color-coding based on correctness, was inspired by TensorFlow’s beginner image classification tutorial: https://www.tensorflow.org/tutorials/keras/classification.
     plt.xlabel(f"{class_names[predicted_label]} {100 * np.max(prediction):.2f}% (True: {class_names[true_label]})", color=color)
# Citation:
# The use of bar color coding to highlight the predicted and true labels was inspired by TensorFlow’s image data visualization techniques: https://www.tensorflow.org/tutorials/keras/classification.
     plt.subplot(1, 2, 2)
     plt.grid(False)
     plt.xticks(range(10))
     plt.yticks([])
     bar_plot = plt.bar(class_names, prediction, color="#777777")
     plt.ylim([0, 1])
     bar_plot[predicted_label].set_color('red')
     bar_plot[true_label].set_color('blue')

if __name__ == "__main__":
     main()
