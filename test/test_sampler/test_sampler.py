import unittest
import sys
import cv2

sys.path.append("./src")
import sampler


class TestSampler(unittest.TestCase):
    def test_sampler(self):
        s = sampler.Sampler("./test/test_sampler/test.mp4", 2)
        c = 0
        frame = s.sample()
        while frame is not None:
            c += 1
            frame = s.sample()
        # Assert that the number of extracted frames is equal to round(duration*KPS)
        cap = cv2.VideoCapture("./test/test_sampler/test.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        seconds = duration % 60
        self.assertEqual(c, round(seconds) * 2 + 1)


if __name__ == "__main__":
    unittest.main()
