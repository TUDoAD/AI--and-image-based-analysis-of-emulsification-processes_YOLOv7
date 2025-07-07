import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic

from model import Model

class Controller(QMainWindow):
    model_request = QtCore.pyqtSignal(dict)
    
    def __init__(self):
        super(Controller, self).__init__()
        uic.loadUi('view.ui', self)
        self.show()

        # Connect buttons to corresponding actions
        self.Start_Video_Button.clicked.connect(self.start_video_evaluation)
        self.Start_Image_Button.clicked.connect(self.start_image_evaluation)
        self.Stop_Video_Button.clicked.connect(self.stop_evaluation)
        self.Stop_Image_Button.clicked.connect(self.stop_evaluation)

        # Set progress bar to 0
        self.progressBar.setValue(0)

        # Initialize the model thread
        self.model = Model()
        self.model_thread = QtCore.QThread()
        self.model.moveToThread(self.model_thread)
        
        # Connect signals
        self.model.data_signal.connect(self.update_progress)
        self.model_request.connect(self.model.mode_selection)

    def get_params(self):
        """
        Get the parameters from the GUI and return them as a dictionary.
        """
        data = {}

        # Validate and retrieve input parameters
        try:
            data["px"] = float(self.px_LineEdit.text())
            data["pixel_length"] = float(self.pxtoum_LineEdit.text())
            data["crop_size"] = int(self.CuttingSize_LineEdit.text())
            data["boarder_size"] = int(self.BoarderSize_LineEdit.text())
            data["FPS"] = int(self.FPS_LineEdit.text())
            data["Evaluated_Points"] = int(self.EvaluatedPoints_LineEdit.text())
            data["Lower_Bound"] = int(self.LowerBound_LineEdit.text())
            data["Upper_Bound"] = int(self.UpperBound_LineEdit.text())
            data["Confidence"] = float(self.Confidence_LineEdit.text())/100
            data["Model"] = self.Model_ComboBox.currentText()

        except ValueError:
            self.wrong_input()
            return None

        # Boolean flags from checkboxes
        data["Evaluation"] = self.Evaluation_CheckBox.isChecked()
        data["crop_image"] = self.CropImage_CheckBox.isChecked()
        data["quality_reduction"] = self.Quality_Reduction_CheckBox.isChecked()
        data["Hough_Circle"] = self.Quality_Reduction_CheckBox.isChecked()

        # Quality settings (only if quality reduction is enabled)
        if data["quality_reduction"]:
            data["quality"] = self.Quality_ComboBox.currentText()

        # File paths
        data["video_input_path"] = self.VideoInputPath_textEdit.toPlainText()
        data["image_input_path"] = self.ImageInputPath_textEdit.toPlainText()

        return data 

    def wrong_input(self):
        """
        Show a message box for invalid input.
        """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Please enter valid input values.")
        msg.setWindowTitle("Invalid Input")
        msg.exec_()

        # Stop the evaluation process if wrong input
        self.stop_evaluation()

    def start_video_evaluation(self):
        """
        Start the video evaluation process.
        """
        parameter = self.get_params()
        if not parameter:
            return  # Return if invalid input
        
        parameter["mode"] = "Video"
        
        # Update UI to reflect process state
        self.Start_Video_Button.setEnabled(False)
        self.Start_Image_Button.setEnabled(False)
        self.Stop_Video_Button.setEnabled(True)
        self.Stop_Image_Button.setEnabled(True)

        # Start the thread and emit signal to begin evaluation
        if not self.model_thread.isRunning():
            self.model_thread.start()
        self.model_request.emit(parameter)

    def start_image_evaluation(self):
        """
        Start the image evaluation process.
        """
        parameter = self.get_params()
        if not parameter:
            return  # Return if invalid input
        
        parameter["mode"] = "Image"
        
        # Update UI to reflect process state
        self.Start_Video_Button.setEnabled(False)
        self.Start_Image_Button.setEnabled(False)
        self.Stop_Video_Button.setEnabled(True)
        self.Stop_Image_Button.setEnabled(True)

        # Start the thread and emit signal to begin evaluation
        if not self.model_thread.isRunning():
            self.model_thread.start()
        self.model_request.emit(parameter)

    def stop_evaluation(self):
        """
        Stop the evaluation process and reset UI.
        """
        # Stop the model thread
        self.model_thread.quit()
        self.model_thread.wait()  # Ensure the thread has stopped

        # Reset UI to reflect stopped state
        self.Start_Video_Button.setEnabled(True)
        self.Start_Image_Button.setEnabled(True)
        self.Stop_Video_Button.setEnabled(False)
        self.Stop_Image_Button.setEnabled(False)

        # Show a message box to inform the user
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The evaluation has been stopped.")
        msg.setWindowTitle("Evaluation Stopped")
        msg.exec_()

    def update_progress(self, data):
        """
        Update the progress bar and status based on signals from the model.
        """
        # Update progress bar
        self.progressBar.setValue(data.get("progress", 0))

        if data["message"] == "Finished":
            # Evaluation finished, reset UI and show completion message
            self.Start_Video_Button.setEnabled(True)
            self.Start_Image_Button.setEnabled(True)
            self.Stop_Video_Button.setEnabled(False)
            self.Stop_Image_Button.setEnabled(False)

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("The evaluation is complete.")
            msg.setWindowTitle("Evaluation Finished")
            msg.exec_()

            self.model_thread.quit()
            self.Folders_Label.setText(" ")
        else:
            # Update status label with progress message
            self.Status_Label.setText(data["message"])

            # Handle folder index and total folders display properly
            folder_name = data.get("folder", "Unknown")  # Default to 'Unknown' if folder name is missing
            folder_index = data.get("folder_index", 0)   # Use folder index for progress (assuming 0-based)
            total_folders = data.get("total_folders", 1)  # Default total to 1 to avoid division errors
            
            self.Folders_Label.setText("Folder: {}/{} ({})".format(folder_index + 1, total_folders, folder_name))

def run():
    app = QtWidgets.QApplication(sys.argv)
    window = Controller()
    app.exec_()
    # Clean up model thread after the app quits
    window.model_thread.quit()
    window.model_thread.wait()

if __name__ == '__main__':
    run()
