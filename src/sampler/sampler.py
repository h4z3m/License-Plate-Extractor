import sys
import cv2


class Sampler:
    """
    A class for sampling frames from a video file.
    """

    def __init__(self, video_path, target_frames):
        """
        Initializes the class instance with the given video path and target frames.

        Parameters:
            video_path (str): The path to the video file.
            target_frames (int): The number of frames to extract from the video.
        """
        self.video_path = video_path
        self.rate = 1 / target_frames
        self.current_frame = 0

    def sample(self):
        """
        Initializes a video capture object using the provided video path.
        Raises a ValueError if the video cannot be opened.
        Continuously extracts frames from the video until the end is reached or an error occurs.
        Increments the current frame count by the specified rate.
        Returns the extracted frame if successful, otherwise returns None.

        :param self: The instance of the class.
        :return: The extracted frame or None.
        """
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
