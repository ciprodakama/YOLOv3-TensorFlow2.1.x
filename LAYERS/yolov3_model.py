from LAYERS.darknet53 import *

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

#upsample, out_shape is obtained from the shape of route1 or route2
#Nearest neighbor interpolation is used to unsample inputs to out_shape
def upsample(inputs, out_shape):

    new_height = out_shape[2]
    new_width = out_shape[1]

    inputs = tf.image.resize(inputs, [new_height, new_width], method='nearest')

    return inputs

#YOLOv3 model
def yolov3(n_classes, model_size, anchors, iou_threshold, confidence_threshold, activation):
    
    #Placeholder Input Tensor
    x = inputs = Input([model_size[0],model_size[1],3]) 
    
    #Backbone
    route1, route2, inputs = darknet53(inputs, activation, name='yolo_darknet')
    
    #Detect1
    route, inputs = yolo_convolution_block(inputs, 512, activation, name='yolo_conv0')
    detect1 = yolo_layer(inputs, n_classes, anchors[6:9], model_size, name='yolo_layer0')
    inputs = convolutional_block(route, 256, 1, activation, name='conv_block0')
    upsample_size = route2.get_shape().as_list()
    inputs = upsample(inputs, upsample_size)
    inputs = tf.concat([inputs,route2], axis=3)

    #Detect2
    route, inputs = yolo_convolution_block(inputs, 256, activation, name='yolo_conv1')
    detect2 = yolo_layer(inputs, n_classes, anchors[3:6], model_size, name='yolo_layer1')
    inputs = convolutional_block(route, 128, 1, activation, name='conv_block1')
    upsample_size = route1.get_shape().as_list()
    inputs = upsample(inputs, upsample_size)
    inputs = tf.concat([inputs,route1], axis=3)
    
    #Detect3
    route, inputs = yolo_convolution_block(inputs, 128, activation, name='yolo_conv2')
    detect3 = yolo_layer(inputs, n_classes, anchors[0:3], model_size, name='yolo_layer2')
    inputs = tf.concat([detect1, detect2, detect3], axis=1)

    #Model Building
    model = Model(x, inputs, name='yolov3')

    #model.summary()

    return model