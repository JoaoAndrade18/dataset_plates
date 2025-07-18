from typing import List
import numpy as np
import pandas as pd
import cv2

class SaveFramesPlates:
    def __init__(self, path: str, frames: List[np.ndarray], plates: List[str]):
        self.path = path
        self.frames = frames
        self.plates_OCR = plates

    def save_csv(self):
        data = {
            # 'video_name': self.path.split('/')[-1].split('.')[0],
            'frame': [frame.tolist() for frame in self.frames],
            'plate_OCR': self.plates_OCR
        }
        df = pd.DataFrame(data)

        df.to_csv(self.path, index=False)

    def detect_plates(self, video: str) -> List[np.ndarray]:
        """
        Detect plates in video using Plate Recognition model. openALPR

        Args:
            video (str): Path to the video file.
        """

        frames = self.open_video(video)
        self.frames = self.frames_to_array(frames)

        results = []



    def frames_array_to_image(self, frames: List[np.ndarray]) -> List[str]:
        """
        Convert frames to images and save them.

        Args:
            frames (List[np.ndarray]): List of frames as numpy arrays.

        Returns:
            List[str]: List of paths to saved images.
        """

        image_paths = []
        for i, frame in enumerate(frames):
            image_path = f"{self.path}/frame_{i}.jpg"
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)

        return image_paths
        
    def frames_to_array(self, frames: cv2.VideoCapture) -> List[np.ndarray]:
        """
        Convert frames to numpy arrays.
    
        Args:
            frames (cv2.VideoCapture): OpenCV VideoCapture object.

        Returns:
            List[np.ndarray]: List of frames as numpy arrays.
        """

        frames = []
        while frames.length < len(self.frames):
            ret, frame = frames.read()
            if ret:
                frames.append(frame)
            else:
                break

        return frames

    def open_video(self, video: str) -> List[cv2.VideoCapture]:
        """
        Recevei video path and open it with opencv to get frames.

        Args:
            video (str): Path to the video file.

        Returns:
            List[np.ndarray]: List of frames as numpy arrays.
        """

        cap = cv2.VideoCapture(video)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

        cap.release()
        cap.close()

        return frames