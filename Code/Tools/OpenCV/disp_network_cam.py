from threading import Thread
import cv2 
import numpy as np

def displayNetworkCamera(link):
    '''
    Displays the network camera in a new window

    Parameters
    ----------
    link : str
        The link to the network camera
    '''
    # read camera from network address
    cap = cv2.VideoCapture(link)

    # show video

    while True:
        # read frame
        ret, frame = cap.read()
        # resize frame to 400x400
        frame = cv2.resize(frame, (1000, 1000))
        # show frame with random numerical name
        # cv2.imshow(str(np.random.randint(0, 100)), frame) 
        cv2.imshow('Network Camera', frame)
        # wait for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # get link to network camera
    link = input('Enter the link to the network camera: ')
    # start thread
    t = Thread(target=displayNetworkCamera, args=(link,))
    t.start()