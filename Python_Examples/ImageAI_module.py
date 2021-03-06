import cv2
import numpy as np
import numpy.ma as ma


import matplotlib.pyplot as plt

# For file manipulation and writing
import os
from os import walk
import time

# Debug Tools
from pprint import pprint

# For easy xml creation
import lxml
from lxml import etree as et
from image_processing import getCoordinatesFromImage

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
    TODO
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

# Creates the file structure for ImageAI
def create_endpoint():
    IMAGE_AI_ROOT_PATH = "ImageAI_rsc/"
    create_dir(IMAGE_AI_ROOT_PATH)
    
    os.chdir(IMAGE_AI_ROOT_PATH) # Changes directory to be within Image_ai
    create_dir("annotations")
    create_dir("images")
    create_dir("input_color_map_images")

def create_dir(dir_name: str) -> None:
    try:
        os.makedirs(dir_name)
    except OSError as exception:
        pass
"""
    [x] (1) Creates the folders for the imageAI training set
    [ ] (2) Uses the images in the ./imageAI/images folder to grab the information needed to create the XML file
        (2a) Grabbing the location of the bounding boxes with the <top left, and bottom right> pixels
    [x] (3) Generates the XML based on the data from the images
    (4) Writes the XML file out for that particular image 
"""


"""
    What are the components of the XML file that's generated?

    [x] <?xml version="1.0"?> 
    [x] <annotation>
    [x]   <folder>{$FOLDER_NAME}</folder>
    [x]   <filename>{$FILENAME}</filename>
    [x]   <path>{$FULL PATH}</path>
    [x]   <source>
    [x]     <database>Unknown</database>
    [x]   </source>
    [x]   <size>
    [x]     <width>{$IMAGE_WIDTH}</width>
    [x]     <height>{$IMAGE_HEIGHT}</height>
    [x]     <depth>{$IMAGE_DEPTH}</depth>
    [x]   </size>
    [x]   <segmented>0</segmented>
    [x]   <object>
    [x]     <name>{$OBJECT_NAME}</name>
    [x]     <pose>Unspecified</pose>
    [x]     <truncated>0</truncated>
    [x]     <difficult>0</difficult>
    [x]     <bndbox>
    [x]       <xmin>{$X_UPPER_LEFT}</xmin>
    [x]       <ymin>{$Y_UPPER_LEFT}</ymin>
    [x]       <xmax>{$X_BOTTOM RIGHT}</xmax>
    [x]       <ymax>{$Y_BOTTOM_RIGHT}</ymax>
    [x]     </bndbox>
    [x] </object>
    [x] </annotation>
"""

"""
    Generates the components of the XML
"""
# User must put in the images by hand into the images directory
def mad_libs_xml(foldername: str, filename: str, full_path: str, IMAGE_WIDTH: int, IMAGE_HEIGHT: int, IMAGE_DEPTH: int, label_list: "Array of coordinate labels"):
    #root = et.Element('xml', version='1.0') # Generates top-level xml tag
    annotation_tag = et.Element('annotation')
    #annotation_tag = et.SubElement(root, 'annotation')
    
    # Generates the three path names
    folder_tag = et.SubElement(annotation_tag, 'folder')
    folder_tag.text = foldername
    
    filename_tag = et.SubElement(annotation_tag, 'filename')
    filename_tag.text = filename

    path_tag = et.SubElement(annotation_tag, 'path')
    path_tag.text = full_path

    source_tag = et.SubElement(annotation_tag, 'source')
    database_tag = et.SubElement(source_tag, 'database')
    database_tag.text = "Unknown"

    # Size section of the XML
    size_tag = et.SubElement(annotation_tag, 'size')

    width_tag = et.SubElement(size_tag, 'width')
    width_tag.text = str(IMAGE_WIDTH)

    height_tag = et.SubElement(size_tag, 'height')
    height_tag.text = str(IMAGE_HEIGHT)

    depth_tag = et.SubElement(size_tag, 'depth')
    depth_tag.text = str(IMAGE_DEPTH)
    
    # Segmented tags
    segmented_tag = et.SubElement(annotation_tag, 'segmented')
    segmented_tag.text = "0"

    add_objects_to_XML(label_list, annotation_tag)
    # object_tag = et.SubElement(annotation_tag, 'object')
    # name_tag = et.SubElement(object_tag, 'name')
    # name_tag.text = object_name
    # pose_tag = et.SubElement(object_tag, 'pose')
    # pose_tag.text = "Unspecified"
    # truncated_tag = et.SubElement(object_tag, 'truncated')
    # truncated_tag.text = "0"
    # difficult_tag = et.SubElement(object_tag, 'difficult')
    # difficult_tag.text = "0"
    # bndbox_tag = et.SubElement(object_tag, 'bndbox')
    # xmin_tag = et.SubElement(bndbox_tag, 'xmin')
    # xmin_tag.text = str(x_min)
    # ymin_tag = et.SubElement(bndbox_tag, 'ymin')
    # ymin_tag.text = str(y_min)
    # xmax_tag = et.SubElement(bndbox_tag, 'xmax')
    # xmax_tag.text = str(x_max)
    # ymax_tag = et.SubElement(bndbox_tag, 'ymax')
    # ymax_tag.text = str(y_max)
    
    return annotation_tag
    #print(et.tostring(root, pretty_print=True).decode("utf-8")) 

def add_objects_to_XML(label_list: "List of coordinate labels", XML_ROOT: "Endpoint to attach label"):
    for label in label_list:
        object_tag = et.SubElement(XML_ROOT, 'object')
        
        object_name, x_min, y_min, x_max, y_max = label

        name_tag = et.SubElement(object_tag, 'name')
        name_tag.text = object_name
        pose_tag = et.SubElement(object_tag, 'pose')
        pose_tag.text = "Unspecified"
        truncated_tag = et.SubElement(object_tag, 'truncated')
        truncated_tag.text = "0"
        difficult_tag = et.SubElement(object_tag, 'difficult')
        difficult_tag.text = "0"
        bndbox_tag = et.SubElement(object_tag, 'bndbox')
        xmin_tag = et.SubElement(bndbox_tag, 'xmin')
        xmin_tag.text = str(x_min)
        ymin_tag = et.SubElement(bndbox_tag, 'ymin')
        ymin_tag.text = str(y_min)
        xmax_tag = et.SubElement(bndbox_tag, 'xmax')
        xmax_tag.text = str(x_max)
        ymax_tag = et.SubElement(bndbox_tag, 'ymax')
        ymax_tag.text = str(y_max)

    pass

def getImageData(_filename: "Last Ending Path to the exact image"):
    #print("getImageData(): " + os.getcwd() + "/" + _filename)
    if (os.path.exists("./" + _filename)):
        image = cv2.imread("./" + _filename)
        HEIGHT, WIDTH, COLOR = image.shape
        return WIDTH, HEIGHT, COLOR
    raise Exception("ERROR: getImageData() Invalid filepath")


def getCoordinates(_filename: str):
    os.chdir("../input_color_map_images") # Get out of images directory
    #print(os.getcwd())
    #COLOR_TRIPLET = np.array([36, 195, 175])
    #bin_img = binary_image(_filename, COLOR_TRIPLET) # Grabs the binary masked value of the image
    #print(bin_img)
    mobs = ["chicken"]
    #print("++++++++++++++++++++++++++++++")
    result = getCoordinatesFromImage(_filename, mobs)
    if (len(result) == 0):
        print("Did not detect item in the image")
        raise Exception("Malmo Color Map did not find a mob")
    #print(result[0])
    #print("-------------------------------")
    return result[0]
    
"""
    @param
        "CENTER": [x_center, y_center, width/2, height/2]
        "COCO": [x_upper_left, y_upper_left, width, height]
        "VOC": [x_upper_left, y_upper_left, x_lower_left, y_lower_left,]
"""
def find_bounding_box_coord(bin_img: "filtered image", format_string: "Either {CENTER, COCO, or VOC" ):
                
    if (bin_img is None):
        raise Exception("Binary Image is of type None")

    # Thus our binary_image begin passed is a empty filter
    if (np.count_nonzero(bin_img) == 0):
        raise Exception("Binary Image map is all 0, thus there is no object there.")

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("Masked Image", bin_img)
    # cv2.waitKey(0)
    
    if (len(contours) == 0):
        raise Exception("Could not find a contour for this image")
    assert(len(contours) != 0)

    max_list = []
    min_list = []
    for pixels in contours:
        max_list.append(np.amax(pixels, axis=0))
        min_list.append(np.amin(pixels, axis=0))

    if (len(max_list) == 0 or len(min_list) == 0):
        print("\tFATAL ERROR: Boundary box argument list size 0")
    assert(len(max_list) != 0)
    assert(len(min_list) != 0)

    max = np.amax(np.array(max_list), axis=0)[0]
    min = np.amin(np.array(min_list), axis=0)[0]


    buf = 4
    x,y,w,h = min[0]-buf, min[1]-buf, max[0]-min[0]+2*buf, max[1]-min[1]+2*buf

    if (format_string == "COCO"):
        return x,y,w,h
    elif (format_string == "VOC"):
        return x, y, x+w, y+h
    elif (format_string == "CENTER"):
        return x+(w/2),y+(h/2),w,h
    else:
        print("ERROR FORMAT_STRING not recognized. Please see comment.")

def binary_image(image_path, _COLOR_TRIPLET):
    if (os.path.exists(image_path)):
        img = cv2.imread("./" + image_path)
    else:
        print("Invalid path being passed to binary_image.")
        return
    # cv2.imshow("Normal image", img)
    # cv2.waitKey(0)
    bin_img = np.empty((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img[row].shape[0]):
            if (img[row,col,:] == _COLOR_TRIPLET).all():
                bin_img[row][col] = 1   #turn chicken white
            else:
                bin_img[row][col] = 0

    # cv2.imshow('binary', bin_img)
    # cv2.waitKey(0)
    return bin_img.astype(np.uint8)


def write_xml_to_file(_filename: str, _XML_NODE: "lxml Node"):
    print(f"Processing {_filename}...")
    #print(et.tostring(root, pretty_print=True).decode("utf-8")) 
    os.chdir("../annotations")   # Go up a level back to the {images, colormaps, annotations}
    tree = et.ElementTree(_XML_NODE)
    tree.write(_filename[:-4] + ".xml")

    

def ImageAI_setup():

    create_endpoint() # Creates the tree structure
    
    os.chdir("./images") # Returns back to the image directory
    dirpath, dirnames, filenames = next(walk(os.getcwd())) # Grabs a list of filenames and the dir_path
    
    # For each image within the images directory  
    print("Creating annotation files...")  
    for _filename in filenames:
        
        # Grabs the parent directory
        xml_ext = "/annotations/" + _filename[:-4] + ".xml"
        parent = os.path.dirname(os.getcwd()) 
        xml_path = parent + xml_ext

        #print(xml_path)
        #if (os.path.exists(xml_path)):
        #    continue
        

        # Data for the mad-libs
        _full_path = dirpath + "/" + _filename
        _IMAGE_WIDTH, _IMAGE_HEIGHT, COLOR = getImageData(_filename)    
        _label_list = getCoordinates(_filename) # Natalia's function
        
        # Constructs the Annotation in XML
        XML_ROOT = mad_libs_xml(foldername="images", filename=_filename, full_path=_full_path, IMAGE_WIDTH=_IMAGE_WIDTH, IMAGE_HEIGHT=_IMAGE_HEIGHT, IMAGE_DEPTH=COLOR, label_list=_label_list)    

        # Writes to the annotation file path
        write_xml_to_file(_filename, XML_ROOT)

        # Brings us back to the images file
        os.chdir("../images")
    print("Done!")

    # Design Question: What should we do about the object_name?
        # As of right now there's only 1 "classification, but we may want to have multiple ones?"
    


    # write XML_ROOT into the file
    # Label the file as a .xml file

    
    
from imageai.Detection.Custom import DetectionModelTrainer
# Trains the model using our dataset
from imageai.Detection.Custom import CustomObjectDetection

def run():
    #TEST_FILE_PATH = "./19.jpg" # Testing file on my local directory
    #EXPECTED_NUMBER_OF_COLORS = 100
    #result_array = process_image_into_color_frequency(TEST_FILE_PATH, EXPECTED_NUMBER_OF_COLORS)
    #print(result_array)
    #FILE_TO_COLOR_IMAGES_DIR = "./video_images/02-11-2021_01-24-45"
    #create_train_txt_file(FILE_TO_COLOR_IMAGES_DIR)

    """
        (1) Load color_map images in the directory 'input_color_map_images"
        (2) Load regular images in the directory 'images'
        (3) Run the ImageAI_module script 
    """
    ImageAI_setup()

    os.chdir("../")

    # detector = CustomObjectDetection()
    # detector.setModelTypeAsYOLOv3()
    # detector.setModelPath("./models/holo-lens-model-deriv/detection_model-ex-025--loss-0000.407.h5")
    # detector.setJsonPath("./json/detection_config.json")
    # detector.loadModel()
    # detections = detector.detectObjectsFromImage(input_image="./images/02-11-2021_22-27-51_352.jpg", output_image_path="./output.jpg")
    # 
    # for detection in detections:
    #     print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    # trainer = DetectionModelTrainer()
    # trainer.setModelTypeAsYOLOv3()
    # trainer.setDataDirectory(data_directory="")
    # print("HELLO WORLD" + os.getcwd())  
    # trainer.setTrainConfig(object_names_array=["chicken"], batch_size=2, num_experiments=50, train_from_pretrained_model="./models/resnet50_coco_best_v2.1.0.h5", )
    # # In the above,when training for detecting multiple objects,
    # #set object_names_array=["object1", "object2", "object3",..."objectz"]
    # trainer.trainModel()


    #metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
    #print(metrics)

    # Detect from the images


if (__name__ == "__main__"):
    run()
