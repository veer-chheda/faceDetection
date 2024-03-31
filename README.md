
# Face Detection Using CNN and OpenCV

A Convolutional Neural Network built using transfer learning with weights from VGG16 and OpenCV for face detection.



## Dataset
With OpenCV, train, test and validation dataset comprising of images of faces was generated. I used Labelme to label to add labels to the dataset.

Albumentations was used for data augmentation to improve the accuracy.
## Model Architecture
Model takes the convolution weights from VGG16. Further there are two Dense layers with 2048 neurons. The first layer is for classification of faces. The second layer is for regression which calculates the dimensions of the bounded box for the label.
