from __future__ import print_function
import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import yarp
import COCO_classes_string as coco_classes
import VOC_classes_string as voc_classes

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = coco_classes.CLASSES

#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def read_yarp_image(inport):

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = np.ones((240, 320, 3), dtype=np.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    # Read the data from the port into the image
    inport.read(yarp_image)
    # display the image that has been read
    #matplotlib.pylab.imshow(img_array)

    return img_array, yarp_image

def kp_detector(img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    return img

def vis_detections(im, class_name, dets, thresh=0.5, fig="preview"):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

        im = im.copy()
        cv2.rectangle(im, (int(bbox[0]),int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 4)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), 0, .7, (0, 255, 0))
        # im_crop = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # im_crop = kp_detector(im_crop)

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    cv2.imshow(fig, im)


# def demo(sess, net, image_name):
# def demo(sess, net, im, fig="preview"):
def demo(sess, net, im, fig="preview", classes=CLASSES):
    """Detect object classes in an image using pre-computed object proposals."""

    # # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    # im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # cv2.imshow(fig, im)

    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH, fig=fig)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--src', dest='src_port', help='Yarp port of source images',
                        default='/icub/camcalib/left/out')
    parser.add_argument('--des', dest='des_port', help='Yarp port of receiver',
                        default='/leftCam')
    parser.add_argument('--usage', dest='gpu_usage', help='GPU memory fraction',
                        default='0.4', type=float)


    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # Initialise YARP
    yarp.Network.init()
    # Create a port and connect it to the iCub simulator virtual camera
    input_port = yarp.Port()
    input_port.open(args.des_port)
    port_connected = True
    if not yarp.Network.connect(args.src_port, args.des_port):
        print('Cannot connect to camera port!')
        port_connected = False

    # cv2 preview window
    # cv2.namedWindow("preview")

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_usage
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess = tf.Session(config=config)
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print('\n\nLoaded network {:s}'.format(args.model))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    #
    #
    # for im_name in im_names:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for data/demo/{}'.format(im_name)
    #     demo(sess, net, im_name)
    #
    # plt.show()

    cv2.namedWindow(args.des_port)

    while port_connected:
        im_arr, _ = read_yarp_image(inport=input_port)
#        cv2_img = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
#        cv2.imshow("preview", im_arr)
        demo(sess, net, im_arr, fig=args.des_port, classes=coco_classes.CLASSES)
        key = cv2.waitKey(20)
        if key == 27: #exit on ESC
            break

    # Cleanup
    input_port.close()
    cv2.destroyWindow(args.des_port)

