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
* The model is retrained from face verification model. Due to copyright reason, the training details and datasets are not available.
* The accuracy is about 90% if only use a single RGB image. It is strongly recommended to use a video frames to test.
* The default strategy is: for continuous 30 frames, if the model detects morn than 15 genuine frames, it outputs 'genuine'.
* The codes are incomplete, i will keep working on it.

## Reference
* To add.
