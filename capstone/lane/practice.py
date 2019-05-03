# Main imports
#import Image
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#matplotlib inline
from skimage import io, filters, data
from importlib import reload
import utils; reload(utils)
from utils import *

calibration_dir = "camera_cal"
test_imgs_dir = "test_images"
output_imgs_dir = "output_images"
output_videos_dir = "output_videos"


cal_imgs_paths = glob.glob(calibration_dir + "/*.jpg")

cal_img_path = cal_imgs_paths[11]
#cal_img = load_image(cal_img_path)
#plt.imshow(cal_img)

#image = Image.open(cal_img_path)
#image.show()

image1= io.imread(cal_img_path)
io.imshow(image1)

cx = 9
cy = 6

def findChessboardCorners(img, nx, ny):
	return cv2.findChessboardCorners(img, (nx, ny), None)

def showChessboardCorners(img, nx, ny, ret, corners):
    c_img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.axis('off')
    plt.imshow(img)

ret, corners = findChessboardCorners(to_grayscale(cal_img), cx, cy)
showChessboardCorners(cal_img, cx, cy, ret, corners)


