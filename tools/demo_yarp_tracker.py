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

CLASSES = coco_classes.CLASSES
# CLASSES = voc_classes.CLASSES

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

    return img_array, yarp_image


def vis_detections(im, class_name, dets, thresh=0.5, fig="preview"):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print('inds :', inds)
    if len(inds) == 0:
        return

    idm = np.argmax(dets[:, -1])
    print('idm :', idm)
    print('tracked_bbox :', dets[idm, :4])
    for i in inds:
        bbox = dets[i, :4].astype(int)
        score = dets[i, -1]

        im = im.copy()
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]-2), 0, .7, (0, 255, 0))

    cv2.imshow(fig, im)
    return dets[idm, :4].astype(int)

def get_best_bbox(dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print('inds :', inds)
    if len(inds) == 0:
        return

    idm = np.argmax(dets[:, -1])
    print('idm :', idm)
    print('tracked_bbox :', dets[idm, :4])

    return dets[idm, :4].astype(int)

def obj_tracking(tracker, im, fig="preview"):

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    # Update tracker
    ok, bbox = tracker.update(im)

    # Draw bounding box
    if ok:
        im = im.copy()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(im, p1, p2, (0, 0, 255), 4)

    cv2.imshow(fig, im)
    return ok

def demo(sess, net, im, fig="preview", classes=CLASSES, tracked_obj="person"):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]

    CONF_THRESH = 0.8
    NMS_THRESH = 0.6
    has_trk_obj = False
    trked_bbox = []
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

    #     print('check')
    #     if cls == tracked_obj:
    #         print('Found tracked object', tracked_obj)
    #         trked_bbox = get_best_bbox(dets, thresh=CONF_THRESH)
    #         print('check 1')
    #         print('trked_bbox', trked_bbox)
    #         if trked_bbox is not None:
    #             print('check 2a')
    #             has_trk_obj = True
    #             return has_trk_obj, trked_bbox
    #         print('check 2b')
    #     print('check 3')
    # print('check 4')
    # return has_trk_obj

        trked_bbox = vis_detections(im, cls, dets, thresh=CONF_THRESH, fig=fig)
        print(cls_ind, cls)
        if cls == tracked_obj:
            print('Found tracked object', tracked_obj)
        if cls == tracked_obj and trked_bbox is not None:
            has_trk_obj = True
            break
    return has_trk_obj, trked_bbox




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
    parser.add_argument('--trk', dest='trk_obj', help='Name of object to track',
                        default='person')

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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_usage
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess = tf.Session(config=config)
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)

    print('\n\nLoaded network {:s}'.format(args.model))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    cv2.namedWindow(args.des_port)

    tracker = cv2.Tracker_create("KCF")
    # tracker = cv2.Tracker_create("GOTURN")

    # Obtain the first frame
    # Detect the bbox of tracked object from the first frame
    # Initialize tracker with first frame and bounding box

    print("tracked_obj", args.trk_obj)
    has_obj = False
    ok = False
    tracked_bbox = ()
    while port_connected and not has_obj:
        im_arr, _ = read_yarp_image(inport=input_port)
        has_obj, tracked_bbox = demo(sess, net, im_arr, fig=args.des_port,
                                     classes=CLASSES,
                                     tracked_obj=args.trk_obj)
        print('tracked_bbox', tracked_bbox)
        if has_obj and tracked_bbox is not None:
            bbox = (tracked_bbox[0], tracked_bbox[1],
                    tracked_bbox[2]-tracked_bbox[0],
                    tracked_bbox[3]-tracked_bbox[1])
            ok = tracker.init(im_arr, bbox)

    while port_connected and ok:
        im_arr, _ = read_yarp_image(inport=input_port)

        # # Update tracker
        # ok, bbox = tracker.update(im_arr)
        #
        # # Draw bounding box
        # if ok:
        #     p1 = (int(bbox[0]), int(bbox[1]))
        #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #     cv2.rectangle(im_arr, p1, p2, (0, 0, 255))
        #
        # cv2.imshow(args.des_port,im_arr)
        ok = obj_tracking(tracker, im_arr, fig=args.des_port)

        key = cv2.waitKey(20)
        if key == 27: #exit on ESC
            break

    # Cleanup
    input_port.close()
    cv2.destroyWindow(args.des_port)

