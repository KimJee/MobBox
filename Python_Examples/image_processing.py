import os
import cv2
import numpy as np
import tarfile

chicken = np.array([36, 195, 175])


def parse_video(video_path, timestamp):

    WANTED_FRAME_PER_SECOND = 10
    FRAME_PARAM = 1000 // WANTED_FRAME_PER_SECOND # 1 frame per second

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


def binary_image(image_path):
    img = cv2.imread(image_path)
    # cv2.imshow("Normal image", img)
    # cv2.waitKey(0)
    bin_img = np.empty((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img[row].shape[0]):
            if (img[row,col,:] == chicken).all():
                bin_img[row][col] = 1   #turn chicken white
            else:
                bin_img[row][col] = 0

    # cv2.imshow('binary', bin_img)
    # cv2.waitKey(0)
    return bin_img.astype(np.uint8)


def find_bounding_box(bin_img, color_image):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    #print(bin_img)
    # cnt = contours[0]
    # print(cnt)
    # x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("Bounding box", img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()

    max_list = []
    min_list = []
    for pixels in contours:
        max_list.append(np.amax(pixels, axis=0))
        min_list.append(np.amin(pixels, axis=0))
    max = np.amax(np.array(max_list), axis=0)[0]
    min = np.amin(np.array(min_list), axis=0)[0]

    # print(max, min)

    buf = 4
    x,y,w,h = min[0]-buf, min[1]-buf, max[0]-min[0]+2*buf, max[1]-min[1]+2*buf

    # img = cv2.imread(color_image)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("Bounding box", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return x+(w/2),y+(h/2),w,h


def scale_bounding_box(x, y, w, h, img_w, img_h):
    return x/img_w, y/img_h, w/img_w, h/img_h


def write_box_to_txt(classification, x, y, w, h, textfile_name):
    file = open(textfile_name, "a")
    file.write(f"{classification} {x} {y} {w} {h}")
    file.close()


def find_all_bounding_boxes(image_dir_path, timestamp):
    text_dir = "./bounding_boxes/" + timestamp
    try:
        os.makedirs(text_dir)
    except OSError as exception:
        pass

    for image in sorted(os.listdir(image_dir_path)):
        if image != ".DS_Store":    # add files that are not images
            bin_img = binary_image(image_dir_path + "/" + image)
            x0, y0, w0, h0 = find_bounding_box(bin_img, "./video_images/" + timestamp + "/" + image)
            x, y, w, h = scale_bounding_box(x0, y0, w0, h0, 860, 480)
            write_box_to_txt(0, x, y, w, h, text_dir + f"/{image[:-4]}.txt")


def center_to_ul_br(boxes,img_w,img_h):
    ul_br_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box[0],box[1],box[2],box[3]
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
        print(box)
        bb = read_box_from_txt(text_path+box)
        if format_string == 'CENTER':
            bb = center_to_ul_br(bb,860,480)
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
        # box = label[i]
        # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        box = pred[i]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # cv2.imshow("Bounding box", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(output_dir + image_name, img)


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


def create_bounding_box_images(timestamp, format_string):
    """
    Create a directory in Python_Examples called pred_bounding_boxes and
    create a directory inside of pred_bounding_boxes named with your timestamp.
    Insert your bounding box predicions text files into the timestamp folder
    before running this function

    :param image_path: video image directory
    :param pred_path: predicted bounding box text file directory
    :param label_path: actual bounding box text file directory
    :param output_path: directory to save images with bounding boxes
    :param format_string: COCO, VOC, or CENTER
    """
    pred = read_all_box_from_txt(f"./pred_bounding_boxes/{timestamp}/", format_string)
    label = read_all_box_from_txt(f"./bounding_boxes/{timestamp}/", format_string)

    image_dir = os.listdir(f"./video_images/{timestamp}/")
    for i in range(len(image_dir)):
        draw_bounding_box(image_dir[i],f"./video_images/{timestamp}/",pred[i],label[i],timestamp)


if __name__ == "__main__":

    create_bounding_box_images("02-12-2021_21-09-06","CENTER")
    merge_images("./bounding_box_images/02-12-2021_21-09-06/","chickentestvideo",10)
