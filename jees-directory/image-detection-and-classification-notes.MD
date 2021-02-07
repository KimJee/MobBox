### DEBUG TOOLS

* [ ] Fix spawn_mob functions to specify x/y/z positions
* [ ] Off-limits mob
  * [ ] Create trap for mob or mob that doesn't move much
* [ ] World reset to remove existing mobs
* Learn basic minecraft commands to spawn objects
* Make the agent go into spectator mode
* Bounding boxes around the difference in area
* OpenCV function to make bounding box: [link](https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html)

### TODO for Saturday

---

* [ ] Create a python script that converts the color-map into preset format
  * Input: Color-bit-map from Malmo's frames
  * Output: a 1x5 array of positions [classname, top_left, bottom_left, top_right, bottom_right]
* [ ] Use the labeled data, plus our custom images to train our custom model [link](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* [ ] Test our model (with new unseen images) and compare the error (error metric still in the works)

### Things for after the status report

---

* Then replicate the entire process with different objects (w/ different colors) in minecraft (like pigs, trees)
* Try using the same object multiple times (clone/duplicity)
* [ ] Have another pre-trained model (that isn't YOLOv.3) so we can compare two different models and the ground truth ("color image set")
* [Stretch] Do it in real-time!

[Color-map .mp4 from Malmo] -> Custom Python Script to convert color info to coordinates -> "training labels"
[Regular video stream .mp4 from Malmo] -> "training images"
[training_labels, training_images] -> PyTorch-YOLOv3 -> Output {bounding boxes, classification}

### Concerns

---

* We don't know how to make our project different from [link](https://kevinjchen.github.io/MCSemanticSegmentation/status.html)
* Our partner has been missing about 3 weeks, and has made it to 2 out of the 6 meetings that we've scheduled.
* We're worried about his participation and we know he's juggling a internship, and had an personal emergency.
* Make sure we are on the right path/process pipeline
  * Current progress: color map, video to picture
  * Our plan: create bounding box labels for images, feed images/labels to model(s)
* Questions
  * How do we use pictures/labels in transfer learning/are we doing this correctly?
  * Should we be further in the project?

### Resources

---

* Professor Office Hour Scheduling: [link](https://calendly.com/sameersingh/office-hours?month=2021-02)
* TA Office Hour Scheduling: [link](https://calendly.com/kolbynottingham/ta-appointment?month=2021-02)

### Finished Items

---

* [x] (Natalia) creating an XML (environment string) that creates a bunch of chickens on the map.
  * For testing purposes if we can downscale, reduce the size of the images
* [x] Integrate the color map from the example (team_reward_test.py) into Phoenix.py put the color map outputs to /videos/ directory
* [x] Create a bash script that uses the python script for all images created by frame-splitting

### Largest Hurdle [Fixed]

* We can't get the 'color mapping' of the minecraft world to show with malmo
* Once we figure out the color-mapping, we can then set up the model we use to do the object detection, so we can use that to generate our own model of the color map. Which we will then test for accuracy.

# Learning about Image Detection & Classification

[Original Starting Link](https://stackabuse.com/object-detection-with-imageai-in-python/)

* ImageAI
  * Python Library
  * Contains RetinaNet YOLOv3, and TinyYOLOv3
  * Works offline
    * easily customizeable

## What is a YOLO Algorithm

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

## Regular Algorithm

* (1) Select regions of interest in an image

* (2) Classify these regions using convolutional neural network

## YOLO ALgorithm

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

* So we're going follow  [link](https://appsilon.com/transfer-learning-introduction/) 
* We figured out that this is just a classifier, and not image localization (boundary boxes). So we didn't go any further.

* What is Keras?
  * Deep learning API on python, on top os tensorflow
* What is Fast.ai?
  * Deep learning library (probably not that useful)

* This [link](https://github.com/eriklindernoren/PyTorch-YOLOv3) allows us to use our own custom images with labeled "annotations" to train our own model.