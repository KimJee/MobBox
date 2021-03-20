---
layout: default
title: Final Report
---

<!-- <iframe width="1061" height="597" src="#"
frameborder="0" allow="accelerometer; autoplay; clipboard-write;
encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->

## Project Summary

---

Our mission is to process JPG images taken from the agents’ perspective and identify specific Minecraft mobs within the image. We not only want to identify what type of mobs there are (classic image prediction problem) but also where the mob is within an image, drawing a bounding box around it (object detection problem). We want our agent to be able to identify different mob species (pigs, cows, chicken, etc) and distinguish between multiple “instances” of the same mobs (e.g. 2 chickens in a single image).

For the longest time, we’ve always wanted machines to be able to see the world as we do. But also for that longest time there wasn’t any good way to generalize “seeing things”. The best procedural method up to that point was using a contrast-based system that could segment the video by a certain contrast threshold. However, even that paradigm has it’s flaws, such as: when contrasts aren’t the main determiner of object boundaries,  the lack of extensibillity of the model object detection on other objects. However with the advent of AI and Machine Learning, we can now create a generic, scalable model that can classify and detect objects effectively and easily. By leveraging the use of cutting edge developments in image processing, we hope to create a model in Minecraft that will truly tsee the game as we do.

* [Link to our source code](https://github.com/KimJee/Mobbox)

## Approach

---

We wanted to use state-of-the-art technologies to try to make the best model for the data we can generate. Luckily, there are many computer vision libraries, frameworks, and online tutorials to support our process. While searching for object detection models, we came across this article: [Link](https://stackabuse.com/object-detection-with-imageai-in-python/) that discusses different types of techniques to recognize and filter an image. Eventually, we came across these models: [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) [ImageAi](https://github.com/OlafenwaMoses/ImageAI)

Or simply put, we use a two-pronged approach to create the best model. From one perspective we use the first library with the same default base-model used in the status report, and from the second perspective we use the ImageAI model with it’s differing base-model and independently trained these two models using their respective libraries, with the same training data to see differences in the result.

We utilized this [YOLO](https://github.com/ultralytics/yolov3) library. YOLO stands for “You Only Look Once” which is a computer vision paradigm for a one-scan, super fast library that trades accuracy for speed. The original link comes from the University of Washington by Joseph Redmon and Ali Farhadi. To summarize their paper, YOLOv3 is a system of convolutional neural networks that has 3 main layers. Each one extracts features at their resolution and tries to predict it given the information passed through the previous layers. After passing through all layers, it tries to use that information to classify the object in the picture and determine its location in a bounding box.

![Visual for CNNs](img/image7.png "Diagram for Proposal")

The idea of “convolution” is to ‘filter’ over an image to create a feature map that summarizes the detected features in the input. Think of each image as a stage and each “convolution” is a spotlight that tries to pick up different details of the stage. For example, one layer could see the actors, another could see the backdrop, and by piecing together these “features,” it blends these ideas to understand what is happening. The authors also upsample feature-rich portions of the image which allows for more information per pixel. In the last leg of prediction, they use k-means clustering to determine the bounding box.

We used the provided YOLOv3 model with transfer learning. YOLOv3 comes with its own weights that define their pretrained model; with transfer learning, we customize their weights by training it further to complete our specific object detection task. In order to customize the model to detect mobs on Minecraft, we first generated a dataset of training and validation images and the images’ corresponding ground truth bounding boxes. These images were created by parsing a video created by the Malmo video producer. Two versions of the video were produced: a normal video of what is seen on Minecraft and a colormap video that blocks each entity/block into different solid colors. The colormap is used to create the ground truth bounding boxes. Using the YOLOv3 weights as a starting point, we then trained our model, modifying their weights to fit our data. Our current model has run a total of 20 epochs on about 300 training images.

![MC Chicken](img/image2.png "Diagram for Proposal")
![Colormap Chicken](img/image3.png "Diagram for Proposal")

## Evaluation

---

We have two quantitative metrics to measure the success of our model:

![2-Grid-Chicken](img/image8.png "Gridded Chicken")
![IOU-Visual](img/image1.png "IOU-Visual")

[Red Box = Ground Truth, Black Box = Prediction]

### Intersection over Union (IoU)

IoU measures the similarity between the predicted bounding box and the ground truth bounding box. It finds the ratio between the intersection between the bounding boxes, or the area of pixels that are in both boxes, and the union of the bounding boxes, or the complete area of pixels covered by both boxes together. To compare two different models (e.g. a model we’ve just trained compared to the baseline or previous model), we run the same test images through each model, generating a text file of the predicted bounding boxes for each image. Using the ground truth and predicted bounding boxes, we calculate the IoU for each image and can compare the values generated by each model.

Example from current model: IoU = 0.8826917247650803
Red box = ground truth, green box = prediction

![Alt Text](img/chicken-model.gif)
![Alt Text](img/chicken-model-improved.gif)

### Mean Average Precision (mAP)

The mAP is a metric for the accuracy of an object detection model. It is represented by the area under the curve when plotting the model’s precision against its recall. A correct prediction occurs if the IoU is greater than a specified threshold (YOLOv3 uses IoU > 0.6). The model’s precision is its proportion of true positives out of all of the images and its recall is the proportion of true positives out of the possible positives.

To qualitatively assess our model, we are able to use OpenCV to draw our bounding boxes on top of the images. We can look at the visual accuracy of the bounding boxes as well as the consistency across many frames of a video (see our video summary).

## New Updates

![Alt Text](img/raw-c-desert.jpg)
![Alt Text](img/raw-all-snow.jpg)

### Challenges & Features

One of the next steps in building our model was providing the ability to extend it to different mobs. Our model can now detect 4 different types of mobs. These four minecraft mobs include, chickens, villagers, cows, and pigs. We choose these four mobs because of the variation in size, shape, and color from the background. However, in extending the model to include these mobs, we came across some new challenges.

![Alt Text](img/raw/03-18-2021_12-34-34_7.jpg)
![Alt Text](img/color/03-18-2021_12-34-34_7.jpg)

[Left: Cute Chicken in the Desert] [Right: Terrible color-choices]

### Color Fixing

First, trouble with colors. By using Malmo’s color mapping utility -- a utility that transforms the minecraft video image, into a uniquely colored world that identifies each entity-- we have been able to generate an necessary , semi-automated system of labeling ground truth. However, the pitfall of this system is the similarity of blocks/mob colors that are of very (nearly) the same color values. Consider the image on the right, the sand color is a tennis-ball green, and the chicken has been mapped to a tennis-ball yellow.

On top of that, the compression created from the image file format creates noisy images that cause problems for our automated-color-map-labeler. Thus, we had to use an n-color filter to threshold color values to minimize the error in our ground-truth labeling.

![A color-map view of the jungle biome](img/color-cv-swamp.jpg)
![Picture of the jungle biome](img/raw-cv-swamp.jpg)

[Left: *A color-map view of the jungle biome*] [Right: *Picture of the jungle biome* ]

### Biome Distractors

Also because of the differences in the biomes, there was much more distractors from the mobs themselves. For instance, in the picture on the bottom left, we see that the environment is filled with trees, with “wood” in a reddish-purple, “leaf-block” the neon-green, and “vine-block” the darker-green. All of these distractors add complexity to our model, which then reduces our confidence in any given image.

![A rose or a pig?](img/jees-labels/detected-03-19-2021_17-12-22_29.jpg)
![Alt Text](img/raw-all-snow.jpg)

[Left: *A rose or a pig?*] [Right: *No labels... really?*]

This is well shown in this (left) image, the ImageAI model has a 98.234% confidence that the actual red rose is a pig. Instead of detecting the actual pig next to the grass. These distractors often fool our model.

### Testing Ideology

In our final model we had trained our model on these different categories of training data:
Biomes: [Plains, None, Swamp, Desert, Forest]
In combination with these mobs: [Cow, Cow-Duplicates, Villagers, Villager-Duplicates, Cow+Villager, Chicken, Pig]

In comparison, our testing set consists of...

Biomes: [Ice Plains, Birch Forest]
With these mob combinations: [ chicken, chicken+cow+pig, chicken+cow+villager, chicken+pig, cow, cow+pig, pig, pig + pig ]

<!-- ![Alt Text](img/raw-c-desert.jpg)
![Alt Text](img/color-c-desert.jpg) -->

### Why we chose the combinations

In our training set, the model was tested on all combinations of cow-villager duplicates, but no combinations with chickens or pigs. We left out this information in the training set because we wanted to explore the ability of our model to generalize information. Could the model generalize seeing chickens and pigs not only as singletons, but as objects that could be bundled together with other mobs?

![Clones?](img/jees-labels/detected-03-19-2021_17-12-22_15.jpg)

*AI correctly identifies two pigs*

Another facet we wanted to explore was in terms of duplicate mobs. As seen above, we only trained on “singleton” pigs, yet we tested with “clones” of pigs. This would help us understand if our model could understand the duplicity of any given mob.

![A cow and a pig in the Ice Plains Biome](img/jees-labels/detected-03-19-2021_13-53-42_6.jpg)

*A cow and a pig in the Ice Plains Biome*

![Chicken in the Birch Forest](img/jees-labels/detected-03-18-2021_13-30-59_24.jpg)

*Chicken in the Birch Forest*

### Why we chose the biomes

Our training biomes consisted of biomes with little distractions (Plains, None), biomes with trees (Swamp, Forest), or the Desert. In order to see if our model could achieve similar results with our testing data, the first biome we experimented with was the Birch Forest. This biome was similar enough to the tree-based biomes in the testing data, but still new enough, due to the new kinds of trees introduced, to be tested upon. This acted as our transitional biome to see if our model had met some of the closer goals.

The next biome we used in testing was the Ice Plains biome. This biome consists mostly of snow, and while it is similar to the plain biomes above, it was still different enough due to the stark contrast of the main biome color, white. This would provide some difficulties for the model, especially for certain types of mobs. The chicken, for example, is mostly a white mob, which would be difficult to contrast against the white-textured snow. This would be a good test to see if our model can differentiate between the subject-- the chicken-- and the background--- the icy plains.

![Almost Perfect](img/jees-labels/detected-03-19-2021_19-28-20_11.jpg)

*An almost perfect labelling*

![Chicken-Villager?](img/jees-labels/detected-03-19-2021_19-26-39_3.jpg)

*A Chicken-Villager Appears*

### Other Problems

A common problem with our detection algorithm was that it draws multiple overlapping bounding boxes on one instance of a mob. But this is probably due to the automated-color-mapping pipeline we have created to generate the ground truth labels, and the lack of training required by a detector this large.

### Results [Jee’s Model]

Below are values received from the ImageAI model after roughly 10 epochs.

Here is some sample data of the ImageAI model working on multiple mobs in the image.

[Image of new photo catching multiple mobs]
[Create a gif of the model working on all of the images on the gif]

[{'model_file': 'ImageAI_rsc/models/final-model-malmo/detection_model-ex-011--loss-0004.496.h5', 
'using_iou': 0.5, 
'using_object_threshold': 0.3, 
'using_non_maximum_suppression': 0.5, 
'average_precision': {'chicken': 0.8836422909545835, 'cow': 0.6636756691843161, 'pig': 0.6809333179065912, 'villager': 0.5585584963470884},
 'evaluation_samples': 245, 'map': 0.6967024435981448}]

From this we can see that the easiest to detect in order was, chickens, cows, pigs, then villagers. I have reason to believe it’s because of the broad contrast in color of the chicken which makes it easier to detect incomparison to the others. But it seems like a villager is either hit or miss. Qualitatively speaking, my model was bad. Out of all of 110 images selected as a testing set, only about 30 of them had labels. And from the 30 that had labels, many were misclassified, double classified, or had the plethora of problems mentioned above.

## Resources Used

### AI/ML Libraries & Models & Code

* [Link to main YOLOv3 algorithm](https://github.com/ultralytics/yolov3)
* [Link to another object detection library](https://github.com/OlafenwaMoses/ImageAI)
* [Link to PyTorch Implementation](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### DevTools

* [Weights & Biases](https://wandb.ai/)
* [Image Extraction with OpenCV](https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)
* [Merging .JPG into .MP4](https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/)

### Research papers & Concept Overviews

* [How to implement your own custom model](https://imageai.readthedocs.io/en/latest/customdetection/)
* [What is transfer learning?](https://appsilon.com/transfer-learning-introduction/)
* [Overview of CNNs](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53?gi=2a61e97f5a9d) 
* [Intro to object detection](https://stackabuse.com/object-detection-with-imageai-in-python/)
* [A research paper about  how YOLOv3 works](https://pjreddie.com/media/files/papers/YOLOv3.pdf) 

### StackOverflow & Tutorials

* [ImageAI Object Detection](https://stackabuse.com/object-detection-with-imageai-in-python/)
* [Evaluating Object Detection Models](https://manalelaidouni.github.io/manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html#precision---recall-and-the-confidence-threshold)
* [Another tutorial about object detection models](https://towardsdatascience.com/chess-rolls-or-basketball-lets-create-a-custom-object-detection-model-ef53028eac7d)
* [Jonathan Hui's blog](https://jonathan-hui.medium.com/)

