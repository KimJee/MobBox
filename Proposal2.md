# Transition from Original Idea to something simplier. 

Input: A screenshot from a minecraft world with a "digit" that is built.

The "digit" must be an image from minecraft (preferrably superflat world with
a high contrast block) that is in the shape of the {letter, number, symbol}
that you want decoded.

Input2: A world where you would have a 2x2 grid of blocks {9 distinct blocks
varying in texture, and color} where you can have a mixture of different blocks
on this "wall", in different positions (within the 2x2 grid).

* Different types of blocks

Output2: A list of which blocks are used in the wall. 

(1) Data Gathering

* Input: Take a video of all of the pre-constructed {digits, symbols, number}
* Output: Split the video into individual frames

(2) Encoding that data (machine interpretable)

* Input: High Resolution Video Frame (1920x1080)
* Ouput: Low Resolution image (100x100)

(3) Apply our ML Algorithms

* (MVP): We can ask a pre-trained model (that is trained on digit recognition)
  what their prediction is
* Create our own ML algorithm 

(4) Decode into human readable

(5) Output


Model v1.0

Paramaters we can adjust to make it more "minecraftian"
* Change the time of day
* Change the weather
* Change the blocks used to construct the image
* Change the environment {not super-flat} 

Output: A minecraft text output with the proper {letter, number, symbol} that
you've created.
