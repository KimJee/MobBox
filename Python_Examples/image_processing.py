import cv2
import numpy as np
import numpy.ma as ma

chicken = np.array([36, 195, 175])

img = cv2.imread("./colour_map_images/02-08-2021_20-14-52/2.jpg")
bin_img = np.empty((img.shape[0], img.shape[1]))
for row in range(img.shape[0]):
    for col in range(img[row].shape[0]):
        if (img[row,col,:] == chicken).all():
            bin_img[row][col] = 1   #turn chicken white
        else:
            bin_img[row][col] = 0

cv2.imshow('binary', bin_img)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(bin_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("grayscale")
#
# # Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
#
# # Finding Contours
# # Use a copy of the image e.g. edged.copy()
# # since findContours alters the image
# contours, hierarchy = cv2.findContours(edged,
# cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
#
# print("Number of Contours found = " + str(len(contours)))
#
# # Draw all contours
# # -1 signifies drawing all contours
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#
# print(contours)
# cv2.imshow('Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()