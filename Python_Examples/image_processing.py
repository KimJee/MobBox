import os
import cv2
import numpy as np
import tarfile

chicken = np.array([36, 195, 175])


def parse_video(video_path, timestamp):

    WANTED_FRAME_PER_SECOND = 1
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
            cap.set(cv2.CAP_PROP_POS_MSEC, count*FRAME_PARAM)  # 8 fps
            success, image = cap.read()
            if success:
                cv2.imwrite(image_dir_path + f"/{timestamp}_{count}.jpg", image)
                count += 1

    return "./colour_map_images/" + timestamp


def merge_images():
    pass


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


def read_box_from_txt_yolov3(text_path):
    boxes = []
    file = open(text_path, "r")
    for line in file:
        boxes.append([float(i) for i in line.split(" ")][1:5])
    file.close()
    return boxes


def draw_bounding_box(image_path, boxes):
    """
    Take str of image path and 2D array of box dimensions
    Each box is represented by [upper left x, upper left y, lower right x, lower right y]
    """
    img = cv2.imread(image_path)
    for box in boxes:
        cv2.rectangle(img,(box[0], box[1]),(box[2], box[3]),(0,255,0),2)
    cv2.imshow("Bounding box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# code from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    print(xA,yA,xB,yB)
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    print(interArea)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    print(float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == "__main__":
    for img in [11,98,200]:
        image_path = f"./test_boxes/02-11-2021_22-27-51_{img}.jpg"
        box_path = f"./test_boxes/02-11-2021_22-27-51_{img}.txt"
        label_path = f"./test_boxes/labels/02-11-2021_22-27-51_{img}.txt"
        pred = center_to_ul_br(read_box_from_txt_yolov3(box_path),860,480)
        label = center_to_ul_br(read_box_from_txt_yolov3(label_path),860,480)
        draw_bounding_box(image_path, pred)
        draw_bounding_box(image_path, label)
        print(intersection_over_union(label[0],pred[0]))

    # find_all_bounding_boxes(image_path,"02-08-2021_20-14-52")