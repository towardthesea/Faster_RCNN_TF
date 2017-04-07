import numpy
import yarp
#import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
import cv


import time

# Initialise YARP
yarp.Network.init()


def yarp_to_python(input_port):
    # Create a port and connect it to the iCub simulator virtual camera
    # input_port = yarp.BufferedPortImageRgb()
    # input_port.open("/python-image-port")
    # yarp.Network.connect("/icub/camcalib/left/out", "/python-image-port")

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = numpy.ones((240, 320, 3), dtype=numpy.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

    # Alternatively, if using Python 2.x, try:
    # yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])

    # Read the data from the port into the image
    input_port.read(yarp_image)

    return img_array, yarp_image


if __name__ == '__main__':

    cv2.namedWindow("preview")
    # cv.NamedWindow("image", cv.CV_WINDOW_AUTOSIZE)


    # input_port = yarp.BufferedPortImageRgb()
    input_port = yarp.Port()
    input_port.open("/python-image-port")
    # yarp.Network.connect("/icub/camcalib/left/out", "/python-image-port")
    yarp.Network.connect("/icubSim/cam/left", "/python-image-port")
    # plt.ion()  # turn on interactive mode
    while True:

        # Create numpy array to receive the image and the YARP image wrapped around it
        # img_array = numpy.ones((240, 320, 3), dtype=numpy.uint8)
        # yarp_image = yarp.ImageRgb()
        # yarp_image.resize(320, 240)
        # yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
        #
        # # Read the data from the port into the image
        # input_port.read(yarp_image)

        # Copy to OpenCV, this uses the old cv rather than cv2
        # cv_img = cv.CreateImageHeader([yarp_image.width(), yarp_image.height()], cv.IPL_DEPTH_8U, 3)
        # cv.SetData(cv_img, yarp_image.tostring())
        # cv.CvtColor(cv_img, cv_img, cv.CV_BGR2RGB)

        img_array, _ = yarp_to_python(input_port)

        cv2_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        cv2.imshow("preview", cv2_img)

        key = cv2.waitKey(20)

        if key == 27:  # exit on ESC
                break

    # Cleanup
    input_port.close()