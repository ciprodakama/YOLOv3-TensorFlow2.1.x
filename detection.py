from UTILS.boxes import *
from UTILS.load_weights import *
from LAYERS.darknet53 import *
from LAYERS.common_layers import *
from LAYERS.yolov3_model import *

LEAKY_RELU = 0.1

ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

MODEL_IMG_SIZE = (416,416)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MAXIMUM_OUTPUT_SIZE = 20

NUM_CLASSES = 80

LABELS_PATH = 'DATASET/coco.names'
WEIGHTS_PATH = 'yolov3.weights'
IMG_PATH  = 'DATASET/dog.jpg'
PRED_PATH = 'DATASET' 


def detect_image(Yolo, image_path, output_path, input_size, CLASSES, iou_threshold, score_threshold):
    original_image      = cv2.imread(IMG_PATH) #load image to be detected as np.array using CV2

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size]) #resize and pad original_img
    image_data = image_data[np.newaxis, ...].astype(np.float32)                      #add 1D to processed_img

    pred_bbox = Yolo.predict(image_data)  #using tf.keras.Model.predict() based on model built
    
    pred_bbox = tf.reshape(pred_bbox[0], (-1, tf.shape(pred_bbox[0])[-1])) #reshaping into 2D 

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)

    bboxes = non_max_suppression(bboxes, NUM_CLASSES, MAXIMUM_OUTPUT_SIZE, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)
    
    image = draw_boxes(original_image, bboxes, CLASSES)

    cv2.imshow("Final Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return image


def main():
    yolo = yolov3(NUM_CLASSES, MODEL_IMG_SIZE, ANCHORS,
            IOU_THRESHOLD, CONFIDENCE_THRESHOLD, LEAKY_RELU)
    
    print("Loading weights...\n")
    load_yolo_weights(yolo, WEIGHTS_PATH)
    print("Loading done!\nCongrats!!!!!\n")
    detection = detect_image(yolo, IMG_PATH, PRED_PATH, MODEL_IMG_SIZE[0], LABELS_PATH, IOU_THRESHOLD, CONFIDENCE_THRESHOLD)

    #save to disk eventually
    cv2.imwrite("detection.jpg", detection)
    print("Image Saved!")

if __name__ == "__main__":
    main()