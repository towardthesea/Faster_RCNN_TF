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

    cv2.namedWindow("preview-l")
    cv2.namedWindow("preview-r")
    # cv.NamedWindow("image", cv.CV_WINDOW_AUTOSIZE)


    # input_port = yarp.BufferedPortImageRgb()
    input_port_l = yarp.Port()
    input_port_r = yarp.Port()
    input_port_l.open("/python-image-port-l")
    input_port_r.open("/python-image-port-r")
    # yarp.Network.connect("/icub/camcalib/left/out", "/python-image-port")
    # yarp.Network.connect("/icubSim/cam/left", "/python-image-port")

    yarp.Network.connect("/icub/camcalib/left/out", "/python-image-port-l")
    yarp.Network.connect("/icub/camcalib/right/out", "/python-image-port-r")
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

        img_array_l, _ = yarp_to_python(input_port_l)
        img_array_r, _ = yarp_to_python(input_port_r)

        cv2_img_l = cv2.cvtColor(img_array_l, cv2.COLOR_BGR2RGB)
        cv2.imshow("preview-l", cv2_img_l)

        cv2_img_r = cv2.cvtColor(img_array_r, cv2.COLOR_BGR2RGB)
        cv2.imshow("preview-r", cv2_img_r)

        key = cv2.waitKey(20)

        if key == 27:  # exit on ESC
            break

    # Cleanup
    input_port_l.close()
    input_port_r.close()
    cv2.destroyAllWindows()