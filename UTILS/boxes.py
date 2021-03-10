import numpy as np
import tensorflow as tf
import cv2
import colorsys
import random

#COCO class names parser
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

#Resizing and Padding of the original image
def image_preprocess(image, target_size):
    
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    return image_paded

#Refining boxes to be considered by the NMS function
def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    
    valid_scale=[0, np.inf] #lower and upper bound values used in step #4

    pred_bbox = np.array(pred_bbox)

    #accesing bbox coord, score & #classes_prob
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    
    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,              #(pred_xy - (pred_wh/2), pred_xy + (pred_wh/2)) -->
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)    # getting bottom left top right coord of pred_bbox

    # 2. (xmin, ymin, xmax, ymax) -> Computed based on original image
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clipping out of range boxes
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discarding invalid boxes using upper & lower bounds
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)) #multiply is applied only to axis=-1 thanks to reduce
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discarding boxes based on score_threshold
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1) #newaxis = None

#Applying NMS using tf.image.non_max_suppression function
def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    
    """
    inputs = [ x_min, y_min, x_max, y_max, score, class ] #refined by postprocess_boxes

    Reshaping input in order to have consistent variables for tf.image.non_max_suppression:
    boxes 	A 2-D float Tensor of shape [num_boxes, 4].
    scores 	A 1-D float Tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes).
    """

    #acquiring classes from inputs
    classes = tf.reshape(inputs[:, 5:] , (-1))
    classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1) #casting to float32 for tf.image.non_max_suppression

    #builidng boxes tensor from inputs and classes
    boxes = tf.concat([inputs[:, :5], classes], axis=-1)
    
    #support array to be returned with final bboxes
    array_bboxes = []

    #iterate over all possible classes of COCO Dataset
    for cls in range(n_classes):
        
        #Check if given cls is within boxes
        mask = tf.equal(boxes[:, 5], cls)
        mask_shape = mask.get_shape()

        #If no classes are present skip, else -->
        if mask_shape.ndims != 0:

            #save rows of boxes based on mask 
            class_boxes = tf.boolean_mask(boxes, mask)
            
            #prepare variables for tf.image.non_max_suppression
            boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
            #transform in 1D 
            boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])

            indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
            
            #using indices access selected bbox from class_boxes
            class_boxes = tf.gather(class_boxes, indices)

            #if tf.gather returnes 0 skip, else -->
            if tf.shape(class_boxes)[0] != 0 :
                #saving to support array
                array_bboxes.append(class_boxes)
    
    #concat into single tensor
    best_bboxes = tf.concat(array_bboxes, axis=0)

    return best_bboxes

def draw_boxes(img, outputs, class_names): #visual representation of model's computed BB

    #parsing class names into dictionary
    classes = read_class_names(class_names)
    
    #iterate over each detection from outputs bboxes
    for detection in outputs:
        #splitting output 
        x1y1, x2y2, score, clss = tf.split(detection, [2,2,1,1], axis=0)

        #casting to int tensor values
        x1y1    = (int(x1y1[0]),int(x1y1[1]))
        x2y2    = (int(x2y2[0]),int(x2y2[1]))
        score   = float(score[0])
        idx     = int(clss[0])
        
        #access label with key index
        class_label = classes[idx]
        
        #draw rectangle of bbox and label
        img = cv2.rectangle(img, x1y1, x2y2, (255,0,0), 2) #draw rectangle bases on new coordinates
        img = cv2.putText(img, '{} {:.4f}'.format(class_label, score),x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        
    return img

