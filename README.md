# Motion-Based Handwriting Recognition and Word Reconstruction

A novel real-time handwriting-to-text system for lower-case English alphabet based on motion sensing and deep learning.

## Introduction
It  is  prevalent  in  today’s  world  for  people  to  write  on  a touch screen with a smart pen, as there is a strong need to digitize handwritten content, in order to make the review and indexingeasier.  However, despite the success of character recognitionon digital devices, requiring a digitizer as the writing surface poses a possibly unnecessary restriction to overcome. In addition, in VR and AR applications, it is also hard to recognize the texts written with the motion sensors. In this study, we propose a solution by using a pen equipped with motion sensor to predict the English letters written by the user. Instead of using image data or on-screen stroke data, we analyze the acceleration and gyroscopic data of the pen using machine learning techniques to classify the characters the user is writing.

## Methods
To tackle this problem, we first collected our own dataset by building the hardware and inviting 20 different users to write lowercase English alphabet with the equipment. The input to our algorithm is the yaw, pitch and roll rotation values of the sensor collected when the user is writing with the pen. After preprocessing the data using feature interpolation, data augmentation, and denoising using an autoencoder, we use the processed data to experiment with 4 different classifers: KNN, SVM, CNN and RNN to predict the labels that indicate which letters the user was writing when the sensor data was collected.

## Experiments and Result
Our current best model using RNN (LSTM) achieves 86.6% accuracy in the random split experiment and 53.6% in the subject split experiment. We reach the conclusion based on our technique and experiments that, albeit having a noisy small dataset, it is possible to achieve high accuracy in handwriting recognition based on rotation sensor data, given the user calibrates the model with its handwriting habits before it makes predictions. 

## Built With

* [Python]() - The primary language used
* [TensorFlow, PyTorch]() - The library used
* [Arduino Uno, MPU-9250 Sensor]() - Hardware

## Dependencies

We recommend using anacdonda for environment setup, and use a virtual environment to install dependencies.

```
conda create -n <env_name> python=3.7
conda activate <env_name>
conda install numpy scikit-learn pandas matplotlib pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Development Team

* [**Wanze (Russell) Xie**](https://github.com/russellxie7)
* [**Yutong (Kelly) He**](https://github.com/KellyYutongHe)
* [**Junshen Kevin Chen**](https://github.com/jkvc)



