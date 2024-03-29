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

'''
CLASSES = ('__background__',
           'firm_name',#1
           'firm_num',#2
           'firm_type',#3
           'firm_address',#4
           'firm_owner',#5
           'firm_time',#6
           'firm_capital',#7
           'firm_deadline',#8
           'firm_scope',#9
           'firm_authority',#10
           'firm_aptime')#11
'''
classes2 =('__background__',
           'firm_num',#1
           'firm_name')#2

classes = ('__background__',
           'firm_num',#1
           'firm_name',#2
           'other_tag')

#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, thresh=0.05):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def detect(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)
    scores, boxes = im_detect(sess, net, im)
   
    im = im[:, :, (2, 1, 0)]

    global CLASSES
    CONF_THRESH = 0.4
    NMS_THRESH = 0.2
    results=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            results.append([bbox[0], bbox[1], bbox[2], bbox[3], cls_ind, score, cls])
            #xmin,ymin,xmax,ymax = result[0], result[1], result[2], result[3]
            #object_num = result[4]   prob = result[5]   object_name = result[6]
        #print results
    results = sorted(results, key = lambda results:results[1])
    return results

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

    args = parser.parse_args()

    return args


def init(model_weights, model_net,target):
    global CLASSES
    if target == ['firm_num', 'firm_name']:
        model_weights = os.path.join(cfg.ROOT_DIR, 'FrcnnModel', '2object', 'VGGnet_fast_rcnn_2objects.ckpt')
        model_net = 'VGG_2_test'
        CLASSES = classes2
    else:
        model_weights = os.path.join(cfg.ROOT_DIR, 'FrcnnModel','3object', 'VGGnet_fast_rcnn_3objects.ckpt')
        model_net = 'VGG_16_test'
        CLASSES = classes
    cfg.TEST.HAS_RPN = True
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    net = get_network(model_net)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, model_weights)
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)
    return sess, net

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    im_names = ['1.jpg', '3.jpg', '5.jpg',
                '7.jpg', '9.jpg']


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name)

    plt.show()

