PROJECT DIRECTORY

    [DATASET] (ref COCO)
        coco.names
        dog.jpg  |  horses.jpg  |  kite.jpg  |  scream.jpg

    [LAYERS]
        common_layers.py
            batch_norm              
            conv_2d_padding 
            convolutional_block        
            darknet_residual        
            yolo_convolution_block         
            yolo_layer              
        darknet53.py  
            darknet53         
        yolov3_model.py
            upsample                
            yolov3
        
    [UTILS]
        boxes.py
            read_class_names
            image_preprocess
            postprocess_boxes
            non_max_suppression
            draw_boxes
        dataset.py
        load_weights.py
            load_yolo_weights

    detection.py
        detect_image
        main   
    
    yolov3.weights 

                   
    
    
                   
