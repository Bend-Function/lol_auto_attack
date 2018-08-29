import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time
import pyautogui as pag
start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import threading
import tensorflow as tf
import zipfile
import cv2
import win32gui, win32ui, win32con, win32api
import multiprocessing as mp
from multiprocessing import Process
from PIL import ImageGrab
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

os.chdir('F:\\AI\\models-master\\research\\object_detection')

# Env setup
# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
# Model preparation
# What model to download.

# 这是我们刚才训练的模型
MODEL_NAME = 'lol_hero_4w'

# 对应的Frozen model位置
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'lol.pbtxt')

# 改成自己例子中的类别数，2
NUM_CLASSES = 1


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def shot():
    # global image_np
    # while True:
    image = ImageGrab.grab()
    image_np = np.array(image)
        # print("shot111")
    return image_np


def mouse(min_x,min_y):
    # while True:
    # if min_x > 0 and min_y > 0:
    #     pag.click(min_x + 10, min_y + 5, button="right")

    if min_x > 0 and min_y > 0:
        pag.click(min_x + 60, min_y +120, button="right")


# Detection

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# 测试图片位置

    # PATH_TO_TEST_IMAGES_DIR = os.getcwd() + '\\test_images2'
    # os.chdir(PATH_TO_TEST_IMAGES_DIR)
    # TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    # PATH_TO_TEST_IMAGES_DIR = 'A:\\'
    # TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'now.jpg')]
    # # Size, in inches, of the output images.
    # IMAGE_SIZE = (20, 12)
    #
    # output_path = ('F:\\AI\\Program\\img_lol\\output\\')
# def test(x):
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # TEST_IMAGE_PATHS = os.listdir(os.path.join(image_folder))
        # os.makedirs(output_image_path + image_folder)
        #
        # image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.

        # global min_x, min_y
        # min_x = 0
        # min_y = 0
        # shot()
        # for t in threads:
        #     t.setDaemon(True)
        #     t.start()
        for _ in range(0,150):
        # def t():
            # global min_x, min_y
            # image_np = pool.map(shot, range(1))
            # image = ImageGrab.grab()
            # image_np = np.array(image)

            s = time.time()


            image_np = shot()


            # image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.


            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)


            e = time.time()
            # plt.imshow(image_np)
            #
            # cv2.imshow(image_np)
            # cv2.waitKey(1)
            # plt.show()
            # for i in range(1,100000):
            #     a = 1
            # plt.close()
            s_boxes = boxes[scores > 0.5]
            s_classes = classes[scores > 0.5]
            # print(s_classes)
            # s_scores = scores[scores > 0.5]
            width,height = 1920,1080
            # screen center : 960,540
            # min = 0
            min_distance = 1000000000000

            min_x = 0
            min_y= 0
            distacne = 1000000000
            for i in range(len(s_classes)):
                if s_classes[i] == 1:
                    ty = s_boxes[i][0] * height  # ymin
                    tx = s_boxes[i][1] * width  # xmin
                    # print(newdata.iloc[0, 1])
                    # print(newdata.iloc[0, 2])
                    distacne = (960 - tx) * (960 - tx) + (540 - ty) * (540 - ty)
                    # 找最小的距离

                    if distacne < min_distance and distacne > 50000:
                        min_distance = distacne
                        min_x = tx
                        min_y = ty

            # if min_x > 0 and min_y > 0:
            #     pag.click(min_x + 10, min_y + 5, button="right")
            print(ty,tx)
            mouse(min_x,min_y)
            # t1 = threading.Thread(target=mouse, args=(min_x,min_y))
            # threads.append(t1)
            # t1.setDaemon(True)
            # t1.start()
            # p.start()
            # p = Process(target=mouse, args=(min_x,min_y))
            # p.start()
            ee = time.time()
            # print("Test:  : ", e - s)
            # print("click:  : ", ee - e)
            # print("all :  : ", ee - s)


        # print(min_x,min_y)
        # return min_x,min_y




# threads = []
# t1 = threading.Thread(target=mouse, args=())
# threads.append(t1)
# t2 = threading.Thread(target=shot, args=())
# threads.append(t2)
# t3 = threading.Thread(target=t, args=())
# threads.append(t3)


# if __name__ == '__main__':

    # min_x = 0
    # min_y = 0

    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()






# if __name__ == '__main__':
    # pool = mp.Pool(1)
    # pool.map(test, range(1))
    # p = Process(target=mouse)
    # test(1)



# def multicore():
#     pool = mp.Pool()
#
#     res = pool.map(test, range(2))
#     # print(res)
#
#
# if __name__ == '__main__':
#     multicore()

# if __name__ == '__main__':
#     test()
    # p = Process(target=shot)
    # c = Process(target=mouse)
    # p.start()

    # time.sleep(1)
    # print('执行主进程的内容了')

