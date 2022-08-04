""" 
NOT WORKING
"""
import cv2

from click_on_image import click_event

def get_calibrate(frame):
    """ 
    Read in image and set the ratio and the different trackers colors

    :param frame: image to be read
    :return: ratio, color_list
    """
	# displaying the image
    cv2.imshow('Frame', frame)

	# setting mouse handler for the image
	# and calling the click_event() function
    cv2.setMouseCallback('Frame', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

	# close the window
    cv2.destroyAllWindows()

    return None

class ColorToPosition():
    def __init__(self, colors:list, ratio:float, SHOW_VIDEO:bool=False, SAVE_DATA:bool=False):
        self.colors = colors
        self.ratio = ratio
        self.SHOW_VIDEO = SHOW_VIDEO
        self.SAVE_DATA = SAVE_DATA

        self.frame = None

        self._get_color_bounds()
        self.start()

    def start(self):
        try:
            Thread(target=self.calc_pos, args=()).start()
            return self
        except Exception as e:
            print(e)
            return None

    def calc_pos(self):
        if not self.frame is None:
            # Calculate the position of the color in the frame
            # and save it in the dataframe
            pass

        else:
            print("No frame")
            pass

    def _get_color_bounds(self):
        """
        Get the color bounds for the color

        :return:
        """
        self.upper_color = None
        self.lower_color = None


if __name__ == "__main__":
    # Get the color bounds for the color
    # and save it in the dataframe
    img = cv2.imread('figures/i_spy_0.jpg', 1)
    get_calibrate(frame=img)