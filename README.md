CIFAR-10 Image Classifier:
--------------------------

1)Project Overview:
--------------------

This project involves building a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to develop a model that accurately classifies these images into one of the 10 categories.

2)Features:
------------
Dataset: CIFAR-10, consisting of 60,000 32x32 color images in 10 classes.
Model Architecture: Convolutional Neural Network (CNN) with three convolutional layers, max pooling, and dense layers.
Evaluation: Performance metrics including accuracy and loss.

3)Requirements:
----------------
Python 3.x
TensorFlow 2.x
Matplotlib
Seaborn
NumPy
Scikit-learn

4)Installation:
----------------
Clone the repository:

git clone https://github.com/Panchadip-128/Deep-Learning-Image_Classifier-with-CIFAR-10.git

5)Install the required packages:
-------------------------------

pip install tensorflow matplotlib seaborn numpy scikit-learn

6)Usage:
---------
Prepare the Dataset: The CIFAR-10 dataset is automatically downloaded by TensorFlow when you run the script.

7)Run the Model:
----------------
python cifar10_classifier.py
This will train the model, evaluate it on the test set, and display the results including accuracy, loss, and visualizations of some predictions.

8)Code Overview:
---------------
cifar10_classifier.py: The main script to load the CIFAR-10 dataset, define the CNN model, train the model, and evaluate its performance. It also includes code to visualize some of the predictions and plot the confusion matrix.

9)Results:
----------
Test Accuracy: 68.93%
Test Loss: 0.9027

Example Visualization
The script will display a few test images with their predicted and true labels.

Confusion Matrix
The script generates a confusion matrix to help understand the classification performance for each class.

10)Contributing:
----------------
Feel free to open issues or submit pull requests. Contributions are welcome!
