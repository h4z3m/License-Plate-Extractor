import sys
import cv2


class Sampler:
    def __init__(self, video_path, target_frames):
        self.video_path = video_path
        self.rate = 1 / target_frames
        self.current_frame = 0

    def sample(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise ValueError("Could not open video")
        # Extract frames
        while True:
            video.set(
                cv2.CAP_PROP_POS_MSEC,
                (self.current_frame * 1000),
            )
            ret, frame = video.read()
            if not ret:
                video.release()
                return None
            self.current_frame += self.rate

            return frame
