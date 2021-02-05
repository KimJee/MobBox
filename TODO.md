
# Todo for later

* [ ] Change our Proposal Website

## Todo for 1/29/2021

* [x] Copy and paste Tutorial 1
* [x] Get Video Recording from Malmo
  * This [link](https://elbruno.com/2017/11/09/vs2017-minecraft-game-interaction-agents-missions-definitions-and-recording-with-projectmalmo/) showed me how to finally get the MP4. Can't believe I that this took me 3 long hours. You would think that Malmo would have a much better interface, and that one call to 'recordMP4' would be the *only* thing it needs to show the video. But you have to call another function (getVideo) to show their video. That's some BS.
  * Also, I wish there was much better documentation. This is such a horrible framework and the some of the critical links are DEAD!
* [x] Rearranging desk to have a second monitor if it can even fit ;|
* [ ] Splice those videos into frames
  * Apparently FFMPEG has a video to image splitting functionality so I'm going to give that a shot.
  * According to this [link](https://www.imore.com/how-extract-images-frame-frame-using-ffmpeg-macos) the command we're looking for is...
    * ffmpeg -i {./video/timestamp}.mpg -r 1/1 $filename%03d.bmp
    * Design Comment: We May want to store our files as .bmp \[bitmap\] files instead of .jpeg files for better quality.
    * OOPS! Okay that's not how you do it.
    * Here's the better [documentation](https://trac.ffmpeg.org/wiki/Create%20a%20thumbnail%20image%20every%20X%20seconds%20of%20the%20video)

### INSTRUCTIONS TO CONVERT .MP4 to JPEG

1. Change Directory to the /video/ directory

2. Extract the .tgz into a .tar file.

3. Extract again, the .tar file into a folder

4. Change Directory into that folder

5. Create an folder \['images'\] for the images

6. Run the command ffmpeg -i video.mp4 -vf fps=0.5 ./images/out%d.png

## Todo as a Group

* [ ] Run it through a 'differentiator'

### Goals for Next Meeting

* We want to determine "what" objects are \[Classification \]
* We also want to determine "where" the objects are in the image, and bound them with "boxes" \[Detection \]
* Research more about object classification & detection
  * Find frameworks (e.g. OpenCV) that will try to bound and classify these images

