import cv2
import os
import numpy as np

class Video_Cutter:

    def create_output_path(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        else:
            output_path = os.path.normpath(output_path + "_new")
            output_path=self.create_output_path(output_path)
        return output_path

    def get_videos(self, folder_path):
        """
        Get all videos in the given folder.
        """
        # Get all files in the folder
        files = os.listdir(folder_path)
        # Get all videos
        videos = []
        for file in files:
            if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov"):
                # Erase the file ending
                file = file.split(".")[0]
                videos.append(file)
        return videos
    
    def get_folder_with_images(self, folder_path):
        """
        Get all folders in the given folder that contain images.
        """
        # Get all folders in the folder
        folders = os.listdir(folder_path)
        # Get all folders with images
        folders_with_images = []
        for folder in folders:
            images = os.listdir(os.path.join(folder_path, folder))
            for image in images:
                if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
                    folders_with_images.append(folder)
                    break
        return folders_with_images
    
    def get_images_in_folder(self, folder_path):
        """
        Get all images in the given folder.
        """
        # Get all images in the folder
        images = os.listdir(folder_path)
        # Get rid of all non-image files
        images = [image for image in images if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png")]
        return images

    def crop_image(self, frame, crop_size):
        """
        Crop the image to the defined size from the middle of the image.
        """
        #TODO: Crop the image to the defined size Check
        # Get the size of the image
        height, width, _ = frame.shape
        # Prevent the crop size to be bigger than the image
        crop_size = int(crop_size)
        if crop_size > height or crop_size > width:
            # Get the smallest size
            crop_size = min(height, width)
        # Get the middle of the image
        middle_x = width // 2
        middle_y = height // 2
        # Get the start and end point of the crop
        start_x = middle_x - crop_size // 2
        end_x = middle_x + crop_size // 2
        start_y = middle_y - crop_size // 2
        end_y = middle_y + crop_size // 2
        # Crop the image
        new_frame = frame[start_y:end_y, start_x:end_x]
        return new_frame
    
    def insert_boarder(self, frame, boarder_size):
        """
        Insert a black boarder to the image so that the image size is divisible by 32 and the lazy run of the network is possible.
        """
        if frame is None:
            raise ValueError("The input image is not valid.")
        if boarder_size == 0 and (frame.shape[0] % 32 == 0 and frame.shape[1] % 32 == 0):
            return frame
        # Get the size of the image
        height, width, _ = frame.shape
        boarder_size = int(boarder_size)
        # Calculate the size of the boarder so that the image size is divisible by 32 and the boarder is at least boarder_size
        boarder_x = 32 - ((height+boarder_size) % 32) + boarder_size
        boarder_y = 32 - ((width+boarder_size) % 32) + boarder_size
        # Create the boarder
        boarder = np.zeros((height + boarder_y, width + boarder_x, 3), np.uint8)
        boarder[boarder_x//2:height+boarder_x//2, boarder_y//2:width+ boarder_y//2] = frame
        return boarder

    def create_text_file(self, folder_path, image_type="jpg"):
        """
        Create a text file with all images in the given folder.
        """
        # Get all images in the folder
        images = os.listdir(folder_path)
        # Create the text file
        with open(os.path.join(folder_path, "images.txt"), "w") as file:
            for image in images:
                if image.endswith(image_type):
                    image_path = os.path.join(folder_path, image)
                    file.write("{}\n".format(image_path))


    def resolution_adjustment(self, frame, set_resolution):
        """
        Adjust the resolution of the image.

        Args:
            frame: The image to reduce the quality (as a NumPy array).
            set_resolution: The resolution to which the image should be adjusted. 
                            This can be a string like '4K' or '1080p', or a tuple (width, height).
        
        Returns:
            The image with adjusted resolution.
        """
        # Dictionary to map common resolution presets to width and height
        resolution_map = {
            '4K': (3840, 2160),
            '1080p': (1920, 1080),
            '720p': (1280, 720),
            '480p': (640, 480)
        }
        
        # If set_resolution is a string like '4K' or '1080p', get the corresponding tuple
        if isinstance(set_resolution, str):
            if set_resolution in resolution_map:
                set_resolution = resolution_map[set_resolution]
            else:
                raise ValueError(f"Unknown resolution preset: {set_resolution}")
        
        # Check if the frame is valid
        if frame is None:
            raise ValueError("The input image is not valid.")
        
        # Resize the image to the new resolution using cv2.resize
        adjusted_frame = cv2.resize(frame, set_resolution, interpolation=cv2.INTER_AREA)
        
        return adjusted_frame




if __name__ == "__main__":
    video_cutter = Video_Cutter()
    # get folder with images
    folder_path = "E:\\03_Image_evaluation_tool\\Test\\Testimages"
    print(video_cutter.get_folder_with_images(folder_path))


