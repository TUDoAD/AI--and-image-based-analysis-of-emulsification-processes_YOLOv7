import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DISABLE_XNNPACK"] = "1"
import platform
import time
import random
import numpy as np
import pandas as pd
import onnxruntime as ort

class YoloDetection:
    def __init__(self, ref_length, pixel_length, confidence=60, threshold=0.5, gpu_usage=False, output_path="output", input_path="input", hough_circle=True):
        # Detection Parameters
        self.gpu_usage = gpu_usage
        self.confidence = confidence
        self.threshold = threshold
        self.num_threads = 2
        self.hough_circle_mode = hough_circle
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.modelPath = os.path.normpath(f"{current_dir}/Models/YOLOv7/YOLOv7tiny.onnx") #YOLOV7tiny --> change to YOLOv7 (if wanted)
        self.session = ort.InferenceSession(self.modelPath, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Image Parameters
        self.ref_length = int(ref_length)
        self.pixel_length = int(pixel_length)
        self.scale = round((self.ref_length / self.pixel_length) * 100 / 1000)

        # Image Path
        self.input_path = os.path.normpath(input_path)
        self.output_path = os.path.normpath(output_path)

    def run_image_detection_multi(self):
        """
        Run detection on images in the input path and save the results to the output path.

        Returns:
        data: A dataframe containing the evaluation data
        """
        # Check if the output path exists, if not create it
        self.check_output_path(self.output_path)
        input_path_bool = self.check_input_path(self.input_path)
        if not input_path_bool:  # If the input path does not exist, stop the function
            print("[INFO] Input path does not exist!")
            return

        # Create a dataframe to store the data
        data = pd.DataFrame(columns=["image_name", "droplet_diameter", "number_droplets", "detection_time", "number_outRange", "median", "mean", "IQR", "stand_deviation", "x_25", "x_75", "min_diameter", "max_diameter", "stand_error", "confidence_score", "threshhold", "ref_length", "pixel_length"])

        # Initialize the system with a black image (Lazy run)
        image_lazy_run = np.zeros((640, 640, 3), np.float32)
        self.image_height, self.image_width, _ = image_lazy_run.shape
        classes, scores, boxes, num_obj, droplet_num = self.predict(image_lazy_run)
        
        # Initialize starting time
        start_time = time.time()
        
        # Read the image from the input path
        for filename in os.listdir(self.input_path):
            # Check if the file is an image
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                continue
            # Read the image
            image = cv2.imread(os.path.join(self.input_path, filename))
            # Shape of the image
            self.image_height, self.image_width, _ = image.shape     
            
            # Evaluate the image for object detection
            inference_starting_time = time.time()
            classes, scores, boxes, num_obj, droplet_num = self.predict(image)
            print(f"[INFO] Inference time for {filename}: {time.time() - inference_starting_time:.2f} seconds")

            # Apply Hough Circle Transform
            if self.hough_circle_mode:
                hough_circle_starting_time = time.time()
                overlay = image.copy()
                Diameter, Droplet_Diameter = self.hough_circle(image, overlay ,boxes, classes, num_obj)
                print(f"[INFO] Hough Circle time for {filename}: {time.time() - hough_circle_starting_time:.2f} seconds")
                alpha = 0.4
                
                image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

            else:
                Droplet_Diameter=[]
                for i in range(num_obj):
                    xmin,ymin,xmax,ymax =boxes[i]
                    h = ymax - ymin
                    w = xmax - xmin
                    Droplet_Diameter.append((w+h)/2)
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 128, 128), 2) 

            # Save txt file with the detected objects
            if filename.endswith('.jpg'):
                with open(os.path.join(self.output_path, filename.replace(".jpg", ".txt")), "w") as f:
                    print("Writing to txt file")
                    for i in range(num_obj):
                        xmin,ymin,xmax,ymax =boxes[i]
                        f.write(f"{int(classes[i])} {int(xmax)} {int(ymax)} {int(xmin)} {int(ymin)} {classes[i]}\n")

            # Save the image with the detected objects
            cv2.imwrite(os.path.join(self.output_path, filename), image)
            # Calculate the detection time
            detection_time = time.time() - start_time
            print(f"[INFO] Detection time for {filename}: {detection_time:.2f} seconds")
            # Save the data to the dataframe
            new_data = self.evaluate_data(Droplet_Diameter, detection_time, filename)
            
            # Save data in a dataframe
            if data.empty:
                data = new_data
            else:
                data = pd.concat([data, new_data], ignore_index=True)

        # Save the data to a file in the output path
        data.to_excel(os.path.join(self.output_path, "evaluation_data.xlsx"), index=False)
        return data
            

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        output = self.session.run([self.output_name], {self.input_name: img_input})[0]
        #output = np.squeeze(output, axis=0)

        if output.shape[0] == 1:
             output = np.squeeze(output, axis=0)

        boxes, scores, classes = [], [], []
        image_h, image_w = image.shape[:2]

        for pred in output:
            batch_id, x_max, y_max, x_min, y_min, score, class_id= pred[:7]
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(score)
            classes.append(class_id)

        num_obj = len(classes)
        droplet_num = classes.count(0)
        print(boxes)
        return classes, scores, boxes, num_obj, droplet_num
    
    def hough_circle(self, img, overlay, boxes, classes, num_obj):
        rad=[]
        Droplet_diameter=[]
        box_diameter = []
        box_height = []
        image_new = img.copy()
        
        for i in range(num_obj):
            class_name= int(classes[i])
        
            if class_name == 0: #allowed_classes(droplet)
                #seperate coorodinates from box
                xmin,ymin,xmax,ymax =boxes[i]
                
                # get the subimage that makes up the bounded region 
                box = image_new[int(ymin):int(ymax), int(xmin):int(xmax)]
                # Save box to file
                cv2.rectangle(overlay, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.imwrite("box.jpg", overlay)
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

    def evaluate_data(self, Droplet_Diameter, detection_time, filename):
        """
        Evaluate the data and return it in a dataframe.

        Args:
            Droplet_Diameter: The list of droplet diameters.
            outRange_num: The number of droplets out of range.
            detection_time: The time it took to detect the droplets.
            filename: The name of the image file.

        Returns:
            new_data: A dataframe containing the evaluated data.
        """
        if len(Droplet_Diameter) > 0:
            median = round(np.median(Droplet_Diameter), 3)
            mean = round(np.mean(Droplet_Diameter), 3)
            IQR = round(np.percentile(Droplet_Diameter, 75) - np.percentile(Droplet_Diameter, 25), 3)
            stand_deviation = round(np.std(Droplet_Diameter), 3)
            x_25 = round(np.percentile(Droplet_Diameter, 25), 3)
            x_75 = round(np.percentile(Droplet_Diameter, 75), 3)
            min_diameter = round(np.min(Droplet_Diameter), 3)
            max_diameter = round(np.max(Droplet_Diameter), 3)
            stand_error = round(stand_deviation / np.sqrt(len(Droplet_Diameter)), 3)
        else:
            median = mean = IQR = stand_deviation = x_25 = x_75 = min_diameter = max_diameter = stand_error = 0

        new_data = pd.DataFrame({"image_name": [filename],
                                 "droplet_diameter": [Droplet_Diameter],
                                 "number_droplets": [len(Droplet_Diameter)],
                                 "detection_time": [round(detection_time, 2)],
                                 "median": [median],
                                 "mean": [mean],
                                 "IQR": [IQR],
                                 "stand_deviation": [stand_deviation],
                                 "x_25": [x_25],
                                 "x_75": [x_75],
                                 "min_diameter": [min_diameter],
                                 "max_diameter": [max_diameter],
                                 "stand_error": [stand_error],
                                 "confidence_score": [self.confidence],
                                 "threshhold": [self.threshold],
                                 "ref_length": [self.ref_length],
                                 "pixel_length": [self.pixel_length]})
        
        return new_data

    def load_model(self):
        """
        Load the YOLO model and the labels.
        """
        print("[INFO] Loading ONNX model...")
        session = ort.InferenceSession(self.modelPath, providers=['CPUExecutionProvider'])  # Or 'CUDAExecutionProvider' if using GPU
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return session, input_name, output_name

    def check_input_path(self, input_path):
        """
        Check if the input path exists.

        Args:
            input_path: The path to the input directory.

        Returns:
            input_path_bool: A boolean indicating if the input path exists.
        """
        input_path_bool = os.path.isdir(input_path)
        if not input_path_bool:
            print("[INFO] Input path does not exist!")
        return input_path_bool

    def check_output_path(self, output_path):
        """
        Check if the output path exists, and create it if it does not.

        Args:
            output_path: The path to the output directory.
        """
        if not os.path.isdir(output_path):
            os.makedirs(output_path)



if __name__ == "__main__":
    yolo = YoloDetection(1, 1000, 60, 0.5, False, "output", "input")
    yolo.run_endless_detection()
