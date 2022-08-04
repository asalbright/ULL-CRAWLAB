import cv2
import numpy as np

def image_from_vid(src=0):
    '''
    Click through a video and caputre an image from a frame.
    '''
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(src)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return None
    while cap.isOpened():
        # Capture one frame
        ret, frame = cap.read()
        # Show the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return frame
        elif cv2.waitKey(0) & 0xFF == ord('c'):
            continue

def save_image(frame, path):
    '''
    Save an image to a path.
    '''
    cv2.imwrite(path, frame)

if __name__ == '__main__':
    file = "figures/final_163_test.mp4"
    frame = image_from_vid(src=file)
    save_image(frame, "figures/final_image.png")