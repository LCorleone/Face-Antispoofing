Face Antispoofing
======
Keras implementation of face antispoofing based on single RGB images.

****
	
|Author|LCorleone|
|---|---
|E-mail|lcorleone@foxmail.com


****
## Requirements
* tensorflow 1.5
* keras 2.2.0
* some common packages like numpy and so on.

## Input
* A video that contains one face. (genuine or fake, from printed or replay attacks)

## Output
* True: genuine face
* False: fake face

## Details
* MTCNN is used to detect faces.
* The base CNN model is resnet100 from [insightface](https://github.com/deepinsight/insightface).
* The pretrained model can be downloaded from BAIDU drive [link](https://pan.baidu.com/s/17VmWbMODFV-ghi3hOgZhOA) password: 4yf9
* The model is retrained from face verification model. Due to copyright reason, **the training details and datasets are not available**.
* The accuracy is about 90% if only use a single RGB image. It is strongly recommended to use videos to test. You can derectly record a video by mobile phone and use another mobile phone to record a replay attack to test.
* The default strategy is: for continuous 30 frames, if the model detects morn than 15 genuine frames, it outputs True.
* **About threshold** For videos captured by iphone, the prediction (score) of genuine face is about 10-4 while the prediction of fake face jumps from 0.1 to 0.99, so i set the threshold at 0.001. The camera may have some effects on the threshold. Thus, print the prediction and feel free to adjust it! However, if the prediction for genuine face changes dramaticly, maybe the model is useless. 
* The codes are incomplete, i will keep working on it. If you have some problems, make a issue!

## Visualization
![left true / right false](https://github.com/LCorleone/Face-antispoofing/tree/master/gif/GIF.gif)

## Reference
* To be added.
