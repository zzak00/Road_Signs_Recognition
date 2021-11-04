
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
camera = PiCamera ()
camera.resolution=(640,480)
camera.framerate=30
def stream ():
    raw_capture=PiRGBArray(camera, size=(640,480))
    time.sleep(0.1)
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port =True):
        image=frame.array
        cv2.imshow("Streaming here",image)
        Key=cv2.waitKey(1) & 0xFF
        raw_capture.truncate(0)
    return
    
    
while(1):
    stream ()