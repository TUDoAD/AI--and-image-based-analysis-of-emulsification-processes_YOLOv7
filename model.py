import time
import os
import cv2
import multiprocessing
import pandas as pd
from dataclasses import dataclass
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
import matplotlib.pyplot as plt

from video_cutter import Video_Cutter
from yolov4 import YoloDetection as YOLOv4_Detection
from yolov7 import YoloDetection as YOLOv7_Detection

progress_data = {"progress": 0, "message": "Processing images...", "folder": 1, "total_folders": 1}

class Model(QObject):
    data_signal = Signal(dict)
    

    def __init__(self):
        super().__init__()
        self.cutter = Video_Cutter()

    @Slot(dict)
    def mode_selection(self, data):
        """
        Selects mode (Image or Video) and processes accordingly.
        """
        if data["mode"] == 'Image':
            self.Image_Evaluation(data)
        elif data["mode"] == 'Video':
            self.Video_Evaluation(data)



    @Slot(dict)
    def Image_Evaluation(self, data):
        """Evaluates images in the specified folder."""
        folders = self.cutter.get_folder_with_images(data["image_input_path"])

        for folder in folders:
            output_path_preevaluation, output_path_evaluation = self.setup_paths(data, folder)

            # Get all images in the folder
            images = self.cutter.get_images_in_folder(os.path.join(data["image_input_path"], folder))

            # Pre-process and save images
            if data["crop_image"] or data["quality_reduction"]:
                self.process_images(data, folder, images, output_path_preevaluation)

            # Create a text file for the pre-processed images
            text_file_path = self.cutter.create_text_file(output_path_preevaluation)

            # Object detection if Evaluation flag is True
            if data["Evaluation"]:
                print("Start Object Detection")
                self.object_detection(data, output_path_preevaluation, output_path_evaluation)

        # Signal completion
        progress_data["progress"] = 100
        progress_data["message"] = "Finished"

        self.data_signal.emit(progress_data)

        progress_data["message"] = ""

    def process_images(self, data, folder, images, output_path_preevaluation):
        """
        Helper function to process images.
        This function processes a list of images by performing optional quality reduction,
        cropping, and border insertion. It then saves the processed images to the specified
        output path and updates the progress.
        Args:
            data (dict): A dictionary containing configuration options such as image input path,
                         quality reduction flag, quality value, crop image flag, crop size, and border size.
            folder (str): The folder containing the images to be processed.
            images (list): A list of image filenames to be processed.
            output_path_preevaluation (str): The path where the processed images will be saved.
        Returns:
            None
        """
        
        progress_data["message"] = f"Processing images in folder {folder}..."
        for idx, image in enumerate(images):
            image_path_final = os.path.join(data["image_input_path"], folder, image)
            frame = cv2.imread(image_path_final)

            # Perform optional quality reduction and cropping
            if data["quality_reduction"]:
                frame = self.cutter.resolution_adjustment(frame, data["quality"])
            if data["crop_image"]:
                frame = self.cutter.crop_image(frame, data["crop_size"])
            
            frame = self.cutter.insert_boarder(frame, data["boarder_size"])

            image_name_final, _ = os.path.splitext(image)
            cv2.imwrite(os.path.join(output_path_preevaluation, f"{image_name_final}.jpg"), frame)

            # Update progress
            progress_data["progress"] = int((idx / len(images)) * 100)
            self.data_signal.emit(progress_data)

    @Slot(dict)
    def Video_Evaluation(self, data):
        """Evaluates videos by processing each frame at specified intervals."""
        videos = self.cutter.get_videos(data["video_input_path"])
        data["FPS"] = int(data["FPS"]) / int(data["Evaluated_Points"])

        for video in videos:
            # Process video and get output paths
            output_path_preevaluation, output_path_evaluation = self.process_video(data, video)

            # Object detection if Evaluation flag is True
            if data["Evaluation"]:
                print("Start Object Detection")
                self.object_detection(data, output_path_preevaluation, output_path_evaluation)
            
        # Signal completion
        progress_data["progress"] = 100
        progress_data["message"] = "Finished"
        self.data_signal.emit(progress_data)


    def process_video(self, data, video):
        """Helper function to process individual video."""
        video_path = os.path.join(data["video_input_path"], f"{video}.mp4")
        capture = cv2.VideoCapture(video_path)
        number_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        lower_bound, upper_bound = self.calculate_bounds(data)

        output_path_preevaluation, output_path_evaluation = self.setup_paths(data, video, is_video=True)
        print(f"Processing video: {video}")

        frame_counter = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            rest = frame_counter % int(data["FPS"])
            if self.is_frame_in_bounds(rest, lower_bound, upper_bound):
                frame = self.process_frame(frame, data)
                image_name = f"{video}_{frame_counter}"
                cv2.imwrite(os.path.join(output_path_preevaluation, f"{image_name}.jpg"), frame)

            # Update progress
            self.emit_progress(frame_counter, number_of_frames)
            frame_counter += 1
            time.sleep(0.01)
            

        capture.release()
        cv2.destroyAllWindows()
        return output_path_preevaluation, output_path_evaluation

    def emit_progress(self, current_frame, total_frames):
        """Emit progress update for video processing."""
        progress_data["progress"] = int((current_frame / total_frames) * 100)
        self.data_signal.emit(progress_data)

    def setup_paths(self, data, folder, is_video=False):
        """Set up output paths for pre-evaluation and evaluation."""
        input_type = "Video" if is_video else "Image"
        base_path = data["video_input_path"] if is_video else data["image_input_path"]

        output_path_evaluation = os.path.join(base_path, f"{folder}_Evaluated_{input_type}s")
        output_path_preevaluation = os.path.join(base_path, f"{folder}_Input_{input_type}s")
        self.cutter.create_output_path(output_path_evaluation)
        self.cutter.create_output_path(output_path_preevaluation)

        return output_path_preevaluation, output_path_evaluation

    def calculate_bounds(self, data):
        """Calculate lower and upper bounds for frame processing."""
        fps = int(data["FPS"])
        lower_bound = (fps - int(data["Lower_Bound"])) % fps
        upper_bound = (fps + int(data["Upper_Bound"])) % fps
        return lower_bound, upper_bound

    def is_frame_in_bounds(self, rest, lower_bound, upper_bound):
        """Check if the current frame is within the bounds for processing."""
        if lower_bound != upper_bound:
            if lower_bound > upper_bound:
                if rest >= lower_bound or rest <= upper_bound:
                    return True
            elif lower_bound < upper_bound:
                if rest <= lower_bound or rest > upper_bound:
                    print(f"Rest: {rest}, Lower: {lower_bound}, Upper: {upper_bound}")
                    return True
        else:
            if rest == lower_bound:
                return True

    def process_frame(self, frame, data):
        """Apply cropping and border to the frame if required."""
        if data["quality_reduction"]:
            frame = self.cutter.resolution_adjustment(frame, data["quality"])
        if data["crop_image"]:
            frame = self.cutter.crop_image(frame, data["crop_size"])
        
        frame = self.cutter.insert_boarder(frame, data["boarder_size"])
        
        return frame

    @Slot(dict)
    def object_detection(self, data, output_path_preevaluation, output_path_evaluation):
        """Runs the object detection in a separate process."""
        queue = multiprocessing.Queue()
        print("Model name: ", data["Model"])
        if data["Model"] == "YOLOv4":
            args = (data["pixel_length"], data["px"], data["Confidence"], 0.5, False, output_path_evaluation, output_path_preevaluation, data["Hough_Circle"], queue)
            detection_process = multiprocessing.Process(target=self.run_object_detection_YOLOv4, args=args)
            detection_process.start()
        elif data["Model"] == "YOLOv7":
            args = (data["pixel_length"], data["px"], data["Confidence"], 0.5, False, output_path_evaluation, output_path_preevaluation, data["Hough_Circle"], queue)
            detection_process = multiprocessing.Process(target=self.run_object_detection_YOLOv7, args=args)
            detection_process.start()

        self.monitor_progress(output_path_preevaluation, output_path_evaluation)

        detection_process.join(timeout=10)
        if detection_process.is_alive():
            print("Detection process is still running, terminating process...")
            detection_process.terminate()


        print("Detection process finished")

    def monitor_progress(self, input_path, output_path):
        """Monitors progress of image processing by comparing input and output folders."""
        to_evaluate_images = [img for img in os.listdir(input_path) if img.endswith(".jpg")]
        total_images = len(to_evaluate_images)

        while True:
            evaluated_images = [img for img in os.listdir(output_path) if img.endswith(".jpg")]
            progress_data["progress"] = int((len(evaluated_images) / total_images) * 100)

            self.data_signal.emit(progress_data)

            if len(evaluated_images) >= total_images:
                print("All images evaluated, breakpoint is reached!")
                break

            time.sleep(1)

    @staticmethod
    def run_object_detection_YOLOv4(ref_length, pixel_length, confidence, threshold, gpu_usage, output_path, input_path, hough_circle, queue):
        """Executes YOLOv4 detection in a separate process."""
        detection = YOLOv4_Detection(ref_length, pixel_length, confidence=confidence, threshold=threshold, gpu_usage=gpu_usage, output_path=output_path, input_path=input_path, hough_circle=hough_circle)
        data = detection.run_image_detection()
        queue.put(data)

    @staticmethod
    def run_object_detection_YOLOv7(ref_length, pixel_length, confidence, threshold, gpu_usage, output_path, input_path, hough_circle, queue):
        """Executes YOLOv7 detection in a separate process."""
        detection = YOLOv7_Detection(ref_length, pixel_length, confidence=confidence, threshold=threshold, gpu_usage=gpu_usage, output_path=output_path, input_path=input_path, hough_circle=hough_circle)
        data = detection.run_image_detection_multi()
        queue.put(data)

@dataclass
class input_data:
    mode: str
    image_input_path: str
    video_input_path: str
    Evaluation: bool
    crop_image: bool
    crop_size: int
    quality_reduction: bool
    quality: int
    boarder_size: int
    pixel_length: float
    px: float
    Confidence: float
    FPS: int
    Lower_Bound: int
    Upper_Bound: int
    Evaluated_Points: int


