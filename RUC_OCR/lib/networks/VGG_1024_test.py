import tensorflow as tf
from networks.network import Network


#define
#conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True,padding=DEFAULT_PADDING, group=1, trainable=True)
#max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING)
#roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name)
#def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name)
#lrn(self, input, radius, alpha, beta, name, bias=1.0)
n_classes = 5
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_1024_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
             .conv(7, 7, 96, 2, 2, name='conv1', trainable=False)
             .lrn(5, 0.0005, 0.75, name='norm1', bias = 2.0)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 2, 2, name='conv2')
             .lrn(5, 0.0005, 0.75, name='norm2', bias = 2.0)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 512, 1, 1, name='conv3')
             .conv(3, 3, 512, 1, 1, name='conv4')
             .conv(3, 3, 512, 1, 1, name='conv5'))
        (self.feed('conv5')
             .conv(3,3,256,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2, 1, 1,padding='VALID',relu=False, name='rpn_cls_score'))

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4,1,1,padding='VALID',relu = False,name='rpn_bbox_pred'))

        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TEST', name = 'rois'))
        
        (self.feed('conv5', 'rois')
             .roi_pool(6, 6, 1.0/16, name='pool_5')
             .fc(4096, name='fc6')
             .fc(1024, name='fc7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('fc7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))

