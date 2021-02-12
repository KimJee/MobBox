import cv2
import numpy as np
import numpy.ma as ma

import os
import time
"""
    Returns BGR Order
"""
def map_colors_to_freq(image: "Image from cv2.imread"): 

    result_freq_map = {}    # Create a dictionary

    for row in image:
        for pixel in row:
            # Check to see if that array of pixels is within the dictionary
            #print(pixel)
            #time.sleep(1.0)
            coordinate = tuple( [pixel[0], pixel[1], pixel[2]] )
            result = result_freq_map.setdefault(coordinate, 0) # if key is in the dict, return value, else insert and return 1
            try:
                result_freq_map[coordinate] += 1
            except KeyError:
                print("Key not found in the dictionary.")
                print("Fatal error has occured.")
                exit(-1)

    # Produces the frequency map
    result_freq_map = ({k: value for k, value in sorted(result_freq_map.items(), key=lambda item: item[1], reverse=True)})
    #print(result_freq_map)
    return result_freq_map




def process_image_into_color_frequency(_filepath: str, _expected_unique_colors: int):

    # Try to convert filepath into an image bmp file via cv2
    _img = cv2.imread(_filepath, cv2.IMREAD_COLOR) # Tries to read the image as a regular color file

    if _img is None: # Checks to see if the image was none. 
        print("Image could not be read.")
        exit(-1)

    # Show the image
    freq_color_map = map_colors_to_freq(_img)

    sort_freq_map = sorted(freq_color_map.items(), key=lambda x: x[1], reverse=True)

    result = []
    for idx in range(_expected_unique_colors):
        result.append(sort_freq_map[idx])

    return result

"""
    
"""
def use_kMeans_to_find_median_color():
    pass

"""
    Creates a textfile of 'train.txt'
    Which has the path for all of our data in the format data/custom/image/{date}_{timestamp}.jpg

    @param
        string filepath: the directory of the COLOR_IMAGES 
"""
def create_train_txt_file(filepath: "directory to be parsed through"):

    PREPEND_FILE_PATH = "data/custom/images/"
    output_file = open("train.txt", "w")
    files = os.listdir(filepath) # Creates a list of files
    print(files)
    for _file in files:
        output_file.write(PREPEND_FILE_PATH + _file + "\n")
    #f.write(result_string)

    

    # result_array = []
    # color_keys = []
    # for key in freq_color_map.keys():
    #     color_keys.append(key)
    # #print(f"Colors within the keys from the freq_color_map: {color_keys}")
    # #print(color_keys)
    # for idx in range(_expected_unique_colors):
    #     result_array.append(color_keys[idx])

    # return result_array

def run():
    #TEST_FILE_PATH = "./19.jpg" # Testing file on my local directory
    #EXPECTED_NUMBER_OF_COLORS = 100
    #result_array = process_image_into_color_frequency(TEST_FILE_PATH, EXPECTED_NUMBER_OF_COLORS)
    #print(result_array)
    FILE_TO_COLOR_IMAGES_DIR = "./video_images/02-11-2021_01-24-45"
    create_train_txt_file(FILE_TO_COLOR_IMAGES_DIR)


if (__name__ == "__main__"):
    run()
