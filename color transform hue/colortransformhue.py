__author__ = 'Charlie'
import os, sys, inspect
import cv2
import numpy as np
import argparse

sys.argv = ['colortransformhue', '-s', 'C:\\Users\\a516j\\Downloads\\WhatsApp Image 2024-10-08 at 21.59.47_ca82c59d.jpg', '-t', 'C:\\Users\\a516j\\Downloads\\download.jpeg']

# Info:
# cmd_folder = os.path.dirname(os.path.abspath(__file__))
# __file__ fails if script is called in different ways on Windows
# __file__ fails if someone does os.chdir() before
# sys.argv[0] also fails because it doesn't not always contains the path
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "C:\\Users\\a516j\\OneDrive\\Documents\\pengolahan citra")))

if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils

def image_resize(image, width=-1, height=-1):
    shape = image.shape
    if width == -1:
        if height == -1:
            return image
        else:
            return cv2.resize(image, (int(height * shape[1] / shape[0]), height))
    elif height == -1:
        return cv2.resize(image, (width, int(width * shape[0] / shape[1])))
    else:
        cv2.resize(image, (width, height))

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required = True, help = "Path to source image")
ap.add_argument("-t", "--target", required = True, help = "Path to target image")

args = vars(ap.parse_args())
source = cv2.cvtColor(cv2.imread(args["source"]), cv2.COLOR_BGR2HSV).astype("float32")
target = cv2.cvtColor(cv2.imread(args["target"]), cv2.COLOR_BGR2HSV).astype("float32")

(h,s,v) = cv2.split(source)
(srcLMean, srcLStd) = (h.mean(), h.std())
# (srcVMean, srcVStd) = (v.mean(), v.std())

(h, s, v) = cv2.split(target)
(tarLMean, tarLStd) = (h.mean(), h.std())
# (tarVMean, tarVStd) = (v.mean(), v.std())

#subtract mean value of image
h -= tarLMean
# v -= tarVMean

#scale std deviation based on source image
h *= (tarLStd/srcLStd)
# v *= (tarVStd/srcVStd)

#add source image mean to target
h += srcLMean
# v += srcVMean

# clip the pixel intensities to [0, 255] if they fall outside
# this range
h = np.clip(h, 0, 360)
h *= 0.5

# v += 128
# v= np.clip(v, 0, 255)

# merge the channels together and convert back to the RGB color
# space, being sure to utilize the 8-bit unsigned integer data
# type
transfer = cv2.merge([h, s, v])
transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_HSV2BGR)

cv2.imshow("Color Transform", image_resize(transfer, height = 500))
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", transfer)