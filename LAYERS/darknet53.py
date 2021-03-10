from LAYERS.common_layers import  *

#Darknet53 -> feature extraction
def darknet53(inputs, activation, name=None):
    
    inputs = convolutional_block(inputs, 32, 3, activation)
    inputs = convolutional_block(inputs, 64, 3, activation, strides=2)

    inputs = darknet_residual(inputs, filters=32, activation=activation)

    inputs = convolutional_block(inputs, 128, 3, activation, strides=2)

    for i in range(2):
        inputs = darknet_residual(inputs, filters=64, activation=activation)

    inputs = convolutional_block(inputs, 256, 3, activation, strides=2)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=128, activation=activation)

    route1 = inputs


    inputs = convolutional_block(inputs, 512, 3, activation, strides=2)

    for i in range(8):
        inputs = darknet_residual(inputs, filters=256, activation=activation)

    route2 = inputs


    inputs = convolutional_block(inputs, 1024, 3, activation, strides=2)

    for i in range(4):
        inputs = darknet_residual(inputs, filters=512, activation=activation)

    return route1, route2, inputs

