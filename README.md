# Motion-Based Handwriting Recognition

A novel real-time handwriting-to-text system for lower-case English alphabet based on motion sensing and deep learning.

## Demo Link:
* [Youtube Demo](https://www.youtube.com/watch?v=SGBSVo2U12s}{https://youtu.be/SGBSVo2U12s)

## Introduction
It  is  prevalent  in  today’s  world  for  people  to  write  on  atouch screen with a smart pen, as there is a strong need to dig-itize hand-written content,  to make the review and indexingeasier.  However, despite the success of character recognitionon digital devices, requiring a digitizer as the writingsurface poses a possibly unnecessary restriction to overcome.In addition, in VR and AR applications, it is also hard to rec-ognize the texts written with the motion sensors. We propose a solution by using a pen equipped with motion sensor topredict the English letters written by the user. Instead of usingimage data or on-screen stroke data, we analyze the acceler-ation and gyroscopic data of the pen using machine learning techniques to classify the characters the user is writing.

## Methods
To tackle this problem, we first collected our own dataset by building the hardware and inviting 20 different users to write lowercase English alphabet with the equipment. The input to our algorithm is the yaw, pitch and roll rotation values of the sensor collected when the user is writing with the pen. After preprocessing the data using feature interpolation, data augmentation, and denoising using an autoencoder, we use the processed data to experiment with 4 different classifers: KNN, SVM, CNN and RNN to predict the labels that indicate which letters the user was writing when the sensor data was collected.

## Experiment Result
Our best model using RNN (LSTM) achieves $86.6\%$ accuracy in the random split experiment and $53.6\%$ in the subject split experiment. We reach the conclusion based on our technique and experiments that, albeit having a noisy small dataset, it is possible to achieve high accuracy in handwriting recognition based on rotation sensor data, given the user calibrates the model with its handwriting habits before it makes predictions. See our [report](https://github.com/RussellXie7/DeepMotion/blob/master/paper/DeepMotion_Final_Report.pdf) for more information.


## Built With

* [Python]() - The primary language used
* [TensorFlow, PyTorch]() - The library used
* [Arduino Uno, MPU-9250 Sensor]() - Hardware


## Development Team

* [**Wanze (Russell) Xie**](https://github.com/russellxie7)
* [**Yutong (Kelly )He**](https://github.com/KellyYutongHe)
* [**Junshen Kevin Chen**](https://github.com/CniveK)



