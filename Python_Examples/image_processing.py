import os
import cv2
import numpy as np
import tarfile

sky = np.array([274, 205, 250])
cloud = np.array([224, 195, 181])
dirt = np.array([69, 45, 139])
cow = np.array([105, 7, 1])

MOB = {"chicken": 0, "cow":1}
# COLOR = {"chicken": np.array([36, 195, 175]), "cow": np.array([105, 7, 1])}
COLOR = {"chicken": np.array([36, 195, 175]), "cow": np.array([105, 7, 1])}


def parse_video(video_path, timestamp):

    WANTED_FRAME_PER_SECOND = 2
    FRAME_PARAM = 1000 // WANTED_FRAME_PER_SECOND

    videos = []
    tgz = tarfile.open(video_path, "r:gz")
    tgz.extractall(path="./video")
    for member in tgz.getnames():
        if member.endswith(".mp4"):
            videos.append(member)

    for video in videos:
        if video.endswith("colourmap_video.mp4"):
            image_dir_path = "./colour_map_images/" + timestamp
        else:
            image_dir_path = "./video_images/" + timestamp
        try:
            os.makedirs(image_dir_path)
        except OSError as exception:
            pass

        cap = cv2.VideoCapture("./video/" + video)
        success = True
        count = 0
        while success:
            cap.set(cv2.CAP_PROP_POS_MSEC, count*FRAME_PARAM)
            success, image = cap.read()
            if success:
                image = cv2.resize(image, (430,240))
                cv2.imwrite(image_dir_path + f"/{timestamp}_{count}.jpg", image)
                count += 1

    return "./colour_map_images/" + timestamp


# code based on: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
def merge_images(image_dir,video_name,fps):
    """
    Writes video file with name video_name of images from specified directory
    Images will be sorted into alphanumeric order
    """
    img_array = []
    sorted_dir = sorted(os.listdir(image_dir))
    sorted_dir.sort(key=lambda img:len(img))
    for filename in sorted_dir:
        if filename.endswith(".jpg"):
            img = cv2.imread(image_dir+filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name+'.mp4', fourcc, fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


"""
Functions to generate ground truth bounding boxes from images
"""

def color_threshold_binary(image, color):
    # cv2.imshow('normal', image)
    # cv2.waitKey(0)
    lower_threshold = color - 50
    upper_threshold = color + 50
    mask = cv2.inRange(image, lower_threshold, upper_threshold)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    return mask


def find_bounding_box(bin_img, color_image, format_string):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        _,_,w,h = cv2.boundingRect(contour)
        if w * h >= 25:
            max_pos = np.amax(contour,axis=0)[0]
            min_pos = np.amin(contour,axis=0)[0]

            # print(max, min)

            buf = 5
            x,y = max(0,min_pos[0]-buf), max(0,min_pos[1]-buf)
            w,h = min(429,max_pos[0]+buf)-x, min(239,max_pos[1]+buf)-y

            if format_string == "CENTER":
                boxes.append([(x+(w/2))/430,(y+(h/2))/240,w/430,h/240])
            elif format_string == "VOC":
                boxes.append([x,y,x+w,y+h])
            elif format_string == "COCO":
                boxes.append([x,y,w,h])

            # cv2.rectangle(color_image, (0, 0), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Bounding box", color_image)
            # cv2.waitKey(0)

    return boxes


# def scale_bounding_box(unscaled, img_w, img_h):
#     """ Rescaled bounding boxes to fit YOLO input format (center)"""
#     boxes = []
#     for box in unscaled:
#         boxes.append([box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h])
#     return boxes


def write_box_to_txt(classification, boxes, textfile_name, reset):
    if reset:
        open(textfile_name, 'w').close()
    file = open(textfile_name, "a")
    for box in boxes:
        file.write(f"{classification} {box[0]} {box[1]} {box[2]} {box[3]}\n")
    file.close()


def find_all_bounding_boxes(timestamp, mobs, format_string):
    """
    timestamp: string of timestamp
    mobs: list of strings (of mob types)e.g. ["chicken","cow"]
    format_string: "CENTER", "VOC", or "COCO"
    """
    image_dir_path = "./colour_map_images/" + timestamp
    text_dir = "./bounding_boxes/" + timestamp
    try:
        os.makedirs(text_dir)
    except OSError as exception:
        pass

    for image in os.listdir(image_dir_path):
        if image.endswith(".jpg"):    # add files that are not images
            reset_files = True
            for mob in mobs:
                img = cv2.imread(image_dir_path + "/" + image)
                bin_img = color_threshold_binary(img, COLOR[mob])
                box = find_bounding_box(bin_img, img, format_string)
                write_box_to_txt(MOB[mob], box, text_dir + f"/{image[:-4]}.txt", reset_files)
                reset_files = False


"""
Functions to draw bounding boxes on images
"""


def center_to_ul_br(boxes,img_w,img_h):
    ul_br_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box[0],box[1],box[2],box[3]
        # print(round((x_center - w / 2) * img_w), round((y_center - h / 2) * img_h),
        #     round((x_center + w / 2) * img_w), round((y_center + h / 2) * img_h))
        # print()
        ul_br_boxes.append([round((x_center - w / 2) * img_w), round((y_center - h / 2) * img_h),
                            round((x_center + w / 2) * img_w), round((y_center + h / 2) * img_h)])
    return ul_br_boxes


def read_box_from_txt(text_path):
    boxes = []
    file = open(text_path, "r")
    for line in file:
        boxes.append([float(i) for i in line.split(" ")][1:5])
    file.close()
    return boxes


def read_all_box_from_txt(text_path, format_string):
    boxes = []
    sorted_dir = sorted(os.listdir(text_path))
    sorted_dir.sort(key=lambda img: len(img))
    for box in sorted_dir:
        bb = read_box_from_txt(text_path+box)
        if format_string == 'CENTER':
            bb = center_to_ul_br(bb,430,240)
        boxes.append(bb)
    return boxes


def draw_bounding_box(image_name, image_path, pred, label, timestamp):
    """
    Take str of image path and 2D array of box dimensions
    Each box is represented by [upper left x, upper left y, lower right x, lower right y]
    """
    output_dir = "./bounding_box_images/" + timestamp + "/"
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        pass

    img = cv2.imread(image_path + image_name)
    for i in range(len(label)):
        box = label[i]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        box = pred[i]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # cv2.imshow("Bounding box", img)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    cv2.imwrite(output_dir + image_name, img)


def create_bounding_box_images(timestamp, format_string):
    """
    Create a directory in Python_Examples called pred_bounding_boxes and
    create a directory inside of pred_bounding_boxes named with your timestamp.
    Insert your bounding box prediction text files into the timestamp folder
    before running this function

    format_string: COCO, VOC, or CENTER
    """
    pred = read_all_box_from_txt(f"./pred_bounding_boxes/{timestamp}/", format_string)
    label = read_all_box_from_txt(f"./bounding_boxes/{timestamp}/", format_string)

    image_dir = sorted(os.listdir(f"./video_images/{timestamp}/"))
    image_dir.sort(key=lambda img: len(img))
    for i in range(len(image_dir)):
        draw_bounding_box(image_dir[i],f"./video_images/{timestamp}/",pred[i],label[i],timestamp)


# code from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def intersection_over_union(boxA, boxB):
    """
    Each box is represented by [upper left x, upper left y, lower right x, lower right y]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == "__main__":
    find_all_bounding_boxes("03-04-2021_17-22-05", ["cow","chicken"], "CENTER")
    create_bounding_box_images("03-04-2021_17-22-05", "CENTER")
