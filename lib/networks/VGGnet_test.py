# Comment By Li Jin 20171224
import tensorflow as tf
from networks.network import Network

n_classes = 21
_feat_stride = [16, ]
anchor_scales = [8, 16, 32]

class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        # VGG16 Basic Convution Network used to extract
        # CNN features from conv5_3
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3'))

        # RPN Classification Network
        (self.feed('conv5_3')
         # 3x3 Small Network used to extract "lower-dimensional feature"
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         # Full Connect Network to classify the anchor
         # Three scales and three ratios make k=9 anchors at each point
         # For a 0-1 problem, the output of this layer is 2k
         # This is actually a 0-1 classification problem, we need softmax.
         # So ReLu is not applied here
         # Each output is linked to the specific point's feature
         # So a 1x1 convolution is used to calculate the FC procedure
         .conv(1, 1, len(anchor_scales)*3*2, 1, 1, padding='VALID',
               relu=False, name='rpn_cls_score'))

        # Region proposal bounding box regression
        # As talked before, k=9 anchors are generated
        # Each anchor has 4 parameter: x, y, width, height
        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales)*3*4, 1, 1, padding='VALID', 
               relu=False, name='rpn_bbox_pred'))

        # Softmax for the rpn_cls_score layer
        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob'))

        # Remap it to original size
        (self.feed('rpn_cls_prob')
         .reshape_layer(len(anchor_scales)*3*2, name='rpn_cls_prob_reshape'))

        # Combine all infomation together, generate true proposal
        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))

        # Run Fast RCNN Network to get final result
        (self.feed('conv5_3', 'rois')
         .roi_pool(7, 7, 1.0/16, name='pool_5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

        (self.feed('fc7')
         .fc(n_classes*4, relu=False, name='bbox_pred'))
