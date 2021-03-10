import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, LeakyReLU, Input, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05

#Batch Normalization
def batch_norm(inputs, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON):
    return tf.keras.layers.BatchNormalization(
        axis = 3, momentum = momentum, epsilon = epsilon,
        scale = True, trainable = False)(inputs)

#Convolution with 2-D stride and padding
def conv2d_with_padding(inputs, filters, kernel_size, strides=1):
    if strides == 1:
        padding = 'same'
    else:
        inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)  # top left half-padding
        padding = 'valid'

    output = Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding, use_bias= False, 
                kernel_regularizer=l2(0.0005))(inputs)

    return output

#Convolutional block -> Conv2D + batch_norm + LeakyReLU
def convolutional_block(inputs, filters, kernel_size, activation, name=None, strides=1):
    inputs = conv2d_with_padding(inputs, filters, kernel_size, strides)
    inputs = batch_norm(inputs)
    output = LeakyReLU(alpha=activation)(inputs)
    return output

#Residual block for Darknet 
def darknet_residual(inputs, filters, activation, strides=1):
    
    shortcut = inputs

    inputs = convolutional_block(inputs, filters, 1, activation, strides)
    output = convolutional_block(inputs, 2*filters, 3, activation, strides)

    output = Add()([shortcut, output])

    return output


#Layers to be used after Darknet53 -> yolo_convolution_block and yolo_layer

#Five convolutional blocks for the route, then another one for the output
def yolo_convolution_block(inputs, filters, activation, name=None):
    
    #5 Convolutions
    inputs = convolutional_block(inputs, filters, 1, activation)
    inputs = convolutional_block(inputs, 2*filters, 3, activation)
    inputs = convolutional_block(inputs, filters, 1, activation)
    inputs = convolutional_block(inputs, 2*filters, 3, activation)
    inputs = convolutional_block(inputs, filters, 1, activation)

    #Route
    route = inputs

    inputs = convolutional_block(inputs, 2*filters, 3, activation)

    return route, inputs

#Final detection layer
def yolo_layer(inputs, n_classes, anchors, img_size, name=None):

    n_anchors = len(anchors)

    #Last Convolution
    output = Conv2D(filters=n_anchors * (5 + n_classes),
                    kernel_size=1, strides=1, use_bias=True)(inputs)
    shape_output = output.get_shape().as_list()

    #grid_shapes stores the number of cells for each dimension
    grid_shape = shape_output[1:3]

    output_reshaped = tf.reshape(output, [-1, n_anchors*grid_shape[0]*grid_shape[1], 5+n_classes]) # -->
    #--> For example from (1,13,13,255) to (1,3*13*13,85)
    

    #Now we can get the values of the bounding boxes
    box_centers, box_shapes, confidence, classes = tf.split(output_reshaped, [2,2,1,n_classes], axis=-1)
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    
    '''
    Compute the top-left coordinates for the cells, x_y_offset.
    x_y_offset is an array of couples that has to be added to box_centers.
    Since we have three boxes for each cell, each couple of values that represents the coordinates 
    of a cell has to be repeated for three times. 
    x_y_offset = [[[0. 0.],[0. 0.],[0. 0.],[1. 0.],[1. 0.],[1. 0.],[2. 0.],...,[12. 12.],[12. 12.],[12. 12.]]]
    '''
    x = tf.range(grid_shape[0], dtype=tf.float32) 
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2]) #now x_y_offset has the right shape (as explained above)
    
    '''
    Sigmoid squashes the predicted values in box_centers in a range from 0 to 1,
    because we want to avoid that the center lies in a cell that is not the one considered, 
    after the sum between the predicted x and y coordinates and the top-left coordinates of the cell.
    '''
    box_centers = tf.nn.sigmoid(box_centers)
    #Adding the offset
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    #Computing the actual width and height on the feature map
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

    confidence = tf.nn.sigmoid(confidence)
    #Softmaxing class scores assume that the classes are mutually exclusive
    classes = tf.nn.sigmoid(classes)

    boxes = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    return boxes

    
