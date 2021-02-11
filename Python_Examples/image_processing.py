import os
import cv2
import numpy as np
import tarfile

chicken = np.array([36, 195, 175])


def parse_video(video_path, timestamp):
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
            cap.set(cv2.CAP_PROP_POS_MSEC, count*500)  # 2 fps
            success, image = cap.read()
            if success:
                cv2.imwrite(image_dir_path + f"/{timestamp}_{count}.jpg", image)
                count += 1

    return image_dir_path


def merge_images():
    pass


def binary_image(image_path):

    img = cv2.imread(image_path)
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


def find_bounding_box(bin_img):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)

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
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("Bounding box", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return x,y,w,h


def scale_bounding_box(x, y, w, h, img_w, img_h):
    return x/img_w, y/img_h, w/img_w, h/img_h


def write_boxes_to_txt(classification, x, y, w, h, textfile_name):
    file = open(textfile_name, "a")
    file.write(f"{classification} {x} {y} {w} {h}\n")


def find_all_bounding_boxes(image_dir_path, timestamp):
    text_dir = "./bounding_boxes/" + timestamp
    try:
        os.makedirs(text_dir)
    except OSError as exception:
        pass

    for image in sorted(os.listdir(image_dir_path)):
        if image != ".DS_Store":
            bin_img = binary_image(image_dir_path + "/" + image)
            x0, y0, w0, h0 = find_bounding_box(bin_img)
            x, y, w, h = scale_bounding_box(x0, y0, w0, h0, 860, 480)
            write_boxes_to_txt(0, x, y, w, h, text_dir + f"/{timestamp}_{image[:-4]}.txt")


if __name__ == "__main__":
    image_path = "./colour_map_images/02-08-2021_20-14-52"
    # bin_img = binary_image(image_pat + "/2.jpg")
    # x0,y0,w0,h0 = find_bounding_box(bin_img)
    # x,y,w,h = scale_bounding_box(x0,y0,w0,h0,860,480)
    # print(x,y,w,h)
    # write_boxes_to_txt(0,x,y,w,h,image_path+"/test_chicken.txt")
    find_all_bounding_boxes(image_path,"02-08-2021_20-14-52")