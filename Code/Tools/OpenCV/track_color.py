from video_threaded import *
from image_info import *
import time


class TrackColor():
    def __init__(self, frame=None, colors=None, ratio=None, verbose=False):
        self.verbose = verbose

        self.frame = frame
        self.colors = colors
        self.ratio = ratio
        self.stopped = False

        self._create_mask_bounds()
        self._create_masks()

        self.positions = None   # this needs to be a list of positions eventually

    def start(self):
        Thread(target=self.update_color_pos, args=()).start()
        return self

    def update_color_pos(self):
        while not self.stopped:
            """
            Code here that updates the color position
            """
            for i in range(len(self.masks)):
                # find the color in the mask and find its position
                # self.positions[i] = self.find_color_pos(self.masks[i])
                pass

    def find_color_pos(self, mask):
        pass

    def _create_mask_bounds(self, closeness=5, verbose=False):
        """ 
        Creates two lists, upper and lower, that are bounds of the single list passed in
        """
        # FIXME: This may not not work for colors in the red H value spectrum
        # the color red is in the range of [170, 10] because the H value is a circular spectrum
        upper = []
        lower = []
        for color in self.colors:
            H = np.arange(0, 179)
            up = np.array([color[0] + closeness, 255, 255])
            low = np.array([color[0] - closeness, 100, 100])
            upper.append(up)
            lower.append(low)

        if self.verbose:
            print("Upper Mask: ", upper)
            print("Lower Mask: ", lower)

        return upper, lower

    def _create_masks(self):
        """
        Create masks of the color in the frame
        """
        pass

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get the video path or load in a camera
    path = "figures\Trial.mp4"
    # Start the video capture thread
    video_getter = VideoGetThreaded(path).start()
    # Pick a frame from the video capture thread
    time.sleep(2)
    frame = video_getter.get_frame()
    # Instantiate a ImageInfo object to get the image info
    img_info = ImageInfoGetter(frame)
    colors = img_info.get_img_colors(num_colors=3, color_type="HSV")
    ratio = img_info.get_im_ratio()
    # Delete the image info object as it is no longer needed
    del img_info
    # Instantiate a TrackColor object
    color_tracker = TrackColor(frame, colors, ratio)

    while True:
        # Update the frame inside the TrackColor object with the a frame from the video capture thread
        color_tracker.frame = video_getter.get_frame()
        if cv2.waitKey(1) == ord("q"):
            video_getter.stop()
            color_tracker.stop()
