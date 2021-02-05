### Learning about Image Detection & Classification

[Original Starting Link](https://stackabuse.com/object-detection-with-imageai-in-python/)

* ImageAI
  * Python Library
  * Containes RetinaNet YOLOv3, and TinyYOLOv3
  * Works offline
    * easily customizeable
* Tutorial

[link](https://appsilon.com/object-detection-yolo-algorithm/)

* Image Classification (1 object only)
  * What is there in the image
* Image Localization
  * Drawing a bounding box around the classified object
* Object Detection (multiple objects in picture)
  * What are the objects within an image
  * Classify each object
  * Draw a bounding box around each thing
* Instance Segmentation
  * What is the image
  * Draw a N-gon (polygon) around the boundary of that object

### Regular Algorithm

* (1) Select regions of interest in an image

* (2) Classify these regions using convolutional neural network

### YOLO ALgorithm

* You Only Look Once
  * Tries to draw bounding boxes over the image in one run of the algorithm
  * Single Shot Multibox Detector (for real-time object detection, low accuracy, but pretty fast)
  * Links to
    * [Pytorch](https://github.com/eriklindernoren/PyTorch-YOLOv3)
    * [Fast.ai](https://github.com/Mersive-Technologies/yolov3)
    * [Keras](https://github.com/experiencor/keras-yolo3)
    * Instruction set for using this library called 'arknet'

* I should try using this 'darknet' github repository to see if anything swill be detected
* What is 'Transfer learning'
  * [link](https://appsilon.com/transfer-learning-introduction/)

### List of Questions

* Which ML Framework should we use?
  * Pytorch
  * OpenCV
    * Video -> jpg conversion (instead of ffmpeg)
  * TensorFlow
    * E2E ML Platform
