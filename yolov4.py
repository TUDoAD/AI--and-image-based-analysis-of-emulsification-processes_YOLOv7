import cv2
import os
import platform
import time
import random
import numpy as np
import pandas as pd

class YoloDetection:
    def __init__(self, ref_length, pixel_length, confidence=0.8, threshold=0.5, gpu_usage=True, output_path="output", input_path="input", hough_circle=True):
        # Detection  Parameters
        self.confidence = float(confidence)
        self.threshold = float(threshold)
        self.gpu_usage = True

        print("[INFO] YOLOv4 object detection initialized with the following parameters:")
        print(f"Confidence: {self.confidence}")
        print(f"Threshold: {self.threshold}")
        print(f"GPU Usage: {self.gpu_usage}")


        # Model Parameters
        self.labelsPath = os.path.normpath(os.path.join(os.getcwd(),"Models", "YOLOv4", "obj.names"))
        self.configPath = os.path.normpath(os.path.join(os.getcwd(),"Models", "YOLOv4", "yolov4_1ob.cfg"))
        self.weigthsPath = os.path.normpath(os.path.join(os.getcwd(),"Models", "YOLOv4", "yolov4_1ob_best.weights"))

        # Image Parameters
        self.ref_length = int(ref_length)
        self.pixel_length = int(pixel_length)
        self.scale = round((self.ref_length / self.pixel_length) *100/1000)
        self.hough_circle_mode = hough_circle
        
        # Image Path
        self.input_path = os.path.normpath(input_path)
        self.output_path = os.path.normpath(output_path)

    def run_image_detection(self):
        """
        Run detection on image

        Returns
        data: A dataframe containing the evaluation data
        """
        # Open the labels file
        labels = open(self.labelsPath).read().strip().split("\n")

        # Check if the output path exists, if not create it
        self.check_output_path(self.output_path)
        input_path_bool = self.check_input_path(self.input_path)
        if not input_path_bool: # If the input path does not exist, stop the function
            return

        # Create a dataframe to store the data
        data = pd.DataFrame(columns=["image_name", "droplet_diameter", "number_droplets", "detection_time", "number_outRange", "median", "mean", "IQR", "stand_deviation", "x_25", "x_75", "min_diameter", "max_diameter", "stand_error", "confidence_score", "threshhold", "ref_length", "pixel_length"])
        # Load the YOLO model
        network, labels = self.load_model()
        
        # Initialize the system with a black image (Lazy run)
        image_lazy_run = np.zeros((416,416,3), np.uint8)
        self.image_height, self.image_width, _ = image_lazy_run.shape
        classes, scores, boxes, num_obj, droplet_num, outRange_num = self.predict(network, image_lazy_run, self.confidence, self.threshold)
        # Initialize the Diameter array
        Diameter_array_continous = []
        outrange_continous = 0
        # Initialize starting time
        start_time = time.time()
        # Read the image from the input path
        for filename in os.listdir(self.input_path):
            # Check if the file is an image
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                continue
            # Read the image
            image = cv2.imread(os.path.join(self.input_path, filename))
            # Shape of the image
            self.image_height, self.image_width, _ = image.shape     
            
            overlay = image.copy() 
            
            # Evaluate the image for object detection using YOLOv4
            classes, scores, boxes, num_obj, droplet_num, outRange_num = self.predict(network, image, self.confidence, self.threshold)

            #Apply Hough Circle Transform
            if self.hough_circle_mode:
                Diameter, Droplet_Diameter = self.hough_circle(image, overlay, boxes, classes, num_obj, droplet_num)
                alpha = 0.4
                image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
            else:
                Droplet_Diameter=[]
                for i in range(num_obj):
                    xmin,ymin,w,h =boxes[i]
                    xmax=w+xmin
                    ymax=h+ymin
                    Droplet_Diameter.append((w+h)/2)
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (100, 0, 100), 2)        
                    print(boxes[i])        

            # Save txt file with the detected objects
            if filename.endswith('.jpg'):
                with open(os.path.join(self.output_path, filename.replace(".jpg", ".txt")), "w") as f:
                    print("Writing to txt file")
                    for i in range(num_obj):
                        #print(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                        #xmin, ymin, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                        #print(xmin, ymin, w, h)
                        xmin,ymin,w,h =boxes[i]
                        xmax=w+xmin
                        ymax=h+ymin
                        #x_center = (xmin + (w / 2.0)) / self.image_width
                        #y_center = (ymin + (h / 2.0)) / self.image_height
                        #width = w / self.image_width
                        #height = h / self.image_height
                        f.write(f"{int(classes[i])} {xmin} {ymin} {xmax} {ymax} {scores[i]}\n")


            # Save the image with the detected objects
            cv2.imwrite(os.path.join(self.output_path, filename), image)
            # Calculate the detection time
            detection_time = time.time() - start_time
            #print(f"[INFO] Detection time for {filename}: {detection_time:.2f} seconds") TODO: Check if this is necessary
            # Save the data to the dataframe
            new_data = self.evaluate_data(Droplet_Diameter, outRange_num, detection_time, filename)
            
            # Save data in a dataframe
            if data.empty:
                data = new_data
            else:
                data = pd.concat([data, new_data], ignore_index=True)

        # Save the data to a file in the output path
        data.to_excel(os.path.join(self.output_path, "evaluation_data.xlsx"), index=False)
        print("[INFO] Evaluation data saved successfully")
        return data

    # def save_yolo_annotations(classes, boxes, image_shape, output_dir, filename):
    #     """
    #     Save bounding box annotations in YOLO format.

    #     Args:
    #         classes: List of predicted class IDs.
    #         boxes: List of bounding boxes [x, y, width, height].
    #         image_shape: Tuple of image dimensions (height, width, channels).
    #         output_dir: Directory to save YOLO-format .txt files.
    #         image_filename: Original image filename (used to name the .txt file).
    #     """
    #     image_h, image_w = image_shape[:2]
    #     txt_filename = os.path.splitext(os.path.basename(filename))[0] + ".txt"
    #     txt_path = os.path.join(output_dir, txt_filename)

    #     os.makedirs(output_dir, exist_ok=True)

    #     with open(txt_path, 'w') as f:
    #         for i in range(len(classes)):
    #             class_id = int(classes[i][0]) if isinstance(classes[i], (list, tuple)) else int(classes[i])
    #             x, y, w, h = boxes[i]
    #             x_center = (x + w / 2) / image_w
    #             y_center = (y + h / 2) / image_h
    #             w_norm = w / image_w
    #             h_norm = h / image_h
    #             f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    

    def predict(self, net, image, confidence, threshold):
        """
        Evaluate the image for object detection using YOLOv4.

        Args:
            net: The YOLOv4 detection model.
            image: The input image for prediction.
            confidence: The confidence threshold for object detection.
            threshold: The threshold for non-maximum suppression.

        Returns:
            classes: A list of predicted classes for each detected object.
            scores: A list of confidence scores for each detected object.
            boxes: A list of bounding boxes for each detected object.
            num_obj: The total number of detected objects.
            droplet_num: The number of detected droplets.
            outRange_num: The number of detected objects outside the range.

        """
        image_width, image_height, _ = image.shape
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(image_width,image_height), scale=1/255, swapRB=True) #3552,3552 for 4k; higher resolution 4096,4096
        droplet_num = 0
        outRange_num = 0 

        image_path= self.check_input_path(self.input_path)
        
        classes, scores, boxes = model.detect(image, confidence,threshold)       
        num_obj = int(len(classes))
        for x in classes:
            if x == [0]:  # droplet is 0 and outRange is 1
                droplet_num+=1
            elif x== [1]:
                droplet_num+=1    
            else:
                None
        return classes, scores, boxes, num_obj, droplet_num, outRange_num
    
    def hough_circle(self, img, overlay, boxes, classes, num_obj, droplet_num):
        rad=[]
        Droplet_diameter=[]
        box_diameter = []
        box_height = []
        image_new = img.copy()
        
        for i in range(num_obj):
            class_name= int(classes[i])
        
            if class_name == 0: #allowed_classes(droplet)
                #seperate coorodinates from box
                xmin,ymin,w,h =boxes[i]
                xmax=w+xmin
                ymax=h+ymin
                
                # get the subimage that makes up the bounded region 
                box = image_new[int(ymin):int(ymax), int(xmin):int(xmax)]
                
                #Check if the box is empty
                if box.size == 0:
                    print("Invalid crop dimensions")
                    continue
                try:                    
                    #Convert the box to grayscale
                    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

                    #Apply contrast and brightness to the image
                    clip_hist_percent = 1
                    
                    #Calculate grayscale histogram
                    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
                    hist_size = len(hist)

                    #Calculate cumulative distribution from the histogram
                    accumulator = []
                    accumulator.append(float(hist[0]))
                    for index in range(1, hist_size):
                        accumulator.append(accumulator[index -1] + float(hist[index]))

                    #Locate Points to Clip
                    maximum = accumulator[-1]
                    clip_hist_percent = clip_hist_percent*(maximum/100.0)
                    clip_hist_percent = clip_hist_percent/2.0

                    #Locate left cut
                    minimum_gray = 0
                    while accumulator[minimum_gray] < clip_hist_percent:
                        minimum_gray += 1

                    #Locate right cut
                    maximum_gray = hist_size -1
                    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                        maximum_gray -= 1

                    #Calculate alpha and beta values
                    alpha = 255 / (maximum_gray - minimum_gray)
                    beta = -minimum_gray * alpha
    
                    con_bri = cv2.convertScaleAbs(gray, alpha = alpha, beta = beta)            

                    img_blur = cv2.medianBlur(con_bri, 3)# must be odd number  
                    
                    kernel = np.array ([ [0, -1,  0], 
                                        [-1, 5, -1], 
                                        [0, -1,  0]])
                    img_sharp = cv2.filter2D(img_blur, -1, kernel)
              
                    sf = 4  #scaling factor for resizing the image
                    img_resize = cv2.resize(img_sharp, None, fx=sf, fy=sf, interpolation = cv2.INTER_AREA)
                    
                    diag = np.sqrt(((img_resize.shape[0])**2)+((img_resize.shape[1])**2))  #img_resize
                    rr= int(img_resize.shape[0]/2) #img_resize         
                    circles_img  = cv2.HoughCircles(img_resize, cv2.HOUGH_GRADIENT, dp = 1, minDist = 10000, param1 = 5, param2 = 15, minRadius = rr-10, maxRadius = rr+10)            #img_resize  5 15
                
                except cv2.error as e:
                    print(f"Error converting image to grayscale: {e}")


            if circles_img is None:
                circles_img = [[[0,0,0]]]
            else:
                circles_img  = np.uint16(np.around(circles_img))
                for j in circles_img[0, :]:   
                    k0=((j[0]/sf)+(xmin))
                    k1=((j[1]/sf)+(ymin))
                    k2=(j[2]/sf) 
                    k3=(diag/(sf*2))
                    k4=(img_resize.shape[1]/(sf*2))
                    if k2 in [0]:
                        None
                    else:     
                        
                        r = random.randint(0,255)
                        g = random.randint(0,255)
                        b = random.randint(0,255)

                        cv2.circle(overlay,(int(k0),int(k1)),int(k2),(r,g,b), 2)
                        cv2.circle(img,(int(k0),int(k1)),int(k2),(r,g,b), 2) 
                        cv2.circle(img, (int(k0),int(k1)),int(k2),(r, g, b), -1)
                        #cv2.circle(overlay, (int(k0),int(k1)),int(k2),(0,0,0), 2)
                        #cv2.circle(img,(int(k0),int(k1)),int(k2),(0,0,0), 2) 
                        #cv2.circle(img, (int(k0),int(k1)),int(k2),(0,0,0), -1)
                        cv2.circle(overlay,(int(k0),int(k1)),1 ,(0,0,0),2)
                        cv2.circle(img,(int(k0),int(k1)),1 ,(0,0,0),2)

                    k2 = ((k2*self.ref_length)/self.pixel_length)*1000*2
                    k3 = ((k3*self.ref_length)/self.pixel_length)*1000*2
                    k4 = ((k4*self.ref_length)/self.pixel_length)*1000*2
                    
                    rad.append(k2) 
                    Droplet_diameter.append(k2)
                    box_diameter.append(k3)
                    box_height.append(k4)
                    
        return(rad, Droplet_diameter)
    
    def evaluate_data(self, Diameter_array_continous, outrange_continous,detection_time, filename):
        if len(Diameter_array_continous) == 0:
            Diameter_array_continous = [0]
            # return Empty DataFrame
            new_data = pd.DataFrame({"image_name": filename, "droplet_diameter": [Diameter_array_continous], "number_droplets": 0,
                                            "detection_time": detection_time, "number_outRange": 0, "median": 0, "mean": 0, 
                                            "IQR": 0, "stand_deviation": 0, "x_25": 0, "x_75": 0, "min_diameter": 0, "max_diameter": 0, 
                                            "stand_error": 0, "confidence_score": 0, "threshhold": 0, "ref_length": self.ref_length, 
                                            "pixel_length": self.pixel_length}, index=[0])
        #Calculate the median
        median = np.median(Diameter_array_continous)
        
        #Calculate the mean
        mean = np.mean(Diameter_array_continous)
        
        #Calculate the IQR
        Q1 = np.percentile(Diameter_array_continous, 25)
        Q3 = np.percentile(Diameter_array_continous, 75)
        IQR = Q3 - Q1
        
        #Calculate the standard deviation
        stand_deviation = np.std(Diameter_array_continous)
        
        #Calculate the 25th percentile
        x_25 = np.percentile(Diameter_array_continous, 25)
        
        #Calculate the 75th percentile
        x_75 = np.percentile(Diameter_array_continous, 75)
        
        #Calculate the minimum diameter
        min_diameter = np.min(Diameter_array_continous)
        
        #Calculate the maximum diameter
        max_diameter = np.max(Diameter_array_continous)

        # Calculate the number of droplets
        number_droplets = len(Diameter_array_continous)

        # Calculate the stand error
        stand_error = stand_deviation/np.sqrt(number_droplets)
        
        #Calculate the standard error
        new_data = pd.DataFrame({"image_name": filename, "droplet_diameter": [Diameter_array_continous], "number_droplets": number_droplets, 
                                            "detection_time": detection_time, "number_outRange": outrange_continous, 
                                            "median": median, "mean": mean, "IQR": IQR, "stand_deviation": stand_deviation, 
                                            "x_25": x_25, "x_75": x_75, "min_diameter": min_diameter, "max_diameter": max_diameter, 
                                            "stand_error": stand_error, "confidence_score": self.confidence, "threshhold": self.threshold, 
                                            "ref_length": self.ref_length, "pixel_length": self.pixel_length}, index=[0])
        return new_data

    def load_model(self):
        """
        Load the YOLO model

        Returns
        None
        """
        
        # Check if the weights and configuration files exist
        if not os.path.exists(self.weigthsPath):
            print("[ERROR] Weights file does not exist")
            return None

        if not os.path.exists(self.configPath):
            print("[ERROR] Configuration file does not exist")
            return None
        

        # Open the labels file
        labels = open(self.labelsPath).read().strip().split("\n")

        # Load the YOLO model
        print(self.weigthsPath)
        network = cv2.dnn.readNet(self.weigthsPath, self.configPath)
        print("[INFO] Model loaded successfully")
        if self.gpu_usage:
            # Check if the system is is linux or windows
            if platform.system() == 'Windows':
                print("[INFO] Using GPU for detection")
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            elif platform.system() == 'Linux':
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("[INFO] Using OPENCL for detection")
        else:
            print("[INFO] Using CPU for detection")
        return network, labels
    

    def check_output_path(self, output_path):
        """
        Check if the output path exists, if not create it

        Returns
        None
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print("[INFO] Output path created successfully")
        else:
            print("[INFO] Output path already exists")

    def check_input_path(self, input_path):
        """
        Check if the input path exists

        Returns
        None
        """
        if not os.path.exists(input_path):
            print("[ERROR] Input path does not exist")
            return False
        else:
            print("[INFO] Input path exists")
            return True
        
if __name__ == "__main__":
    ref_length = 1
    pixel_length = 1000
    input_path="W:\Studenten\Derkum\Image_evaluation_tool\Test\Images\CITPAPERYOLOv4"
    output_path="W:\Studenten\Derkum\Image_evaluation_tool\Test\Images\CITPAPERYOLOv4_Evaluated"
    # Initialize the class
    yolo_detection = YoloDetection(ref_length=ref_length, pixel_length=pixel_length, gpu_usage=True, output_path=output_path, input_path=input_path)
    # Run the image detection
    yolo_detection.run_image_detection()
    #yolo_detection.hough_circle()


    

    






        