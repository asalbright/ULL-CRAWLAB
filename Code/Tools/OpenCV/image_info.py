# importing the module
import cv2
import numpy as np

class ImageInfoGetter:
	def __init__(self, frame, verbose=False):
		self.frame = frame
		if self.frame is None:
			self.get_image()
		
		# Get the height and width of the image
		self.height, self.width = self.frame.shape[:2]

		# Set the verbose flag		
		self.verbose = verbose

		self.click_locations = []

	def get_image(self):
		path = input("Enter the path of the image: ") 
		try:
			self.frame = cv2.imread(path)
		except:
			print("Invalid path")
			self.get_image()
	
	def _click_event(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			if self.verbose:
				print(x, y)
			self.click_locations.append((x, y))
	
	def get_im_ratio(self):
		# Create a window to work with
		win_name = "Get the Ratio of the Image" 
		self._create_window(name=win_name)
		# Write on the frame to tell the user what to do
		img = self._write_on_im(self.frame, msg="Select Two Locations", txt_location=(10,50))
		if self.verbose:
			print("Select two locations")
		# displaying the image
		
		while len(self.click_locations) < 2:
			cv2.imshow(win_name, img)
			cv2.waitKey(1)

		cv2.destroyWindow(win_name)

		# get the ratio from the user
		act_dist = input("Enter the actual distance (in): ")
		try:
			act_dist = float(act_dist)
		except:
			print("Invalid input: must be float")
			self.get_im_ratio()
		# Calculate the ratio
		ratio = self._get_pixel_dist(self.click_locations[0], self.click_locations[1]) / act_dist
		if self.verbose:
			print("Ratio (px/in): ", ratio)

		# Clear the click locations
		self.click_locations = []	

		return ratio			
	
	def _get_pixel_dist(self, loc1, loc2):
		# get the distance between the two points
		dist = np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
		return dist

	def get_img_colors(self, num_colors, color_type="HSV"):
		# create a window to work with
		win_name = "Get the Colors of the Markers to Track"
		self._create_window(name=win_name)
		# create a blank list of colors
		colors = []
		# Write on the frame to tell the user what to do
		img = self._write_on_im(self.frame, msg=f"Pick {num_colors} Locations", txt_location=(10,50))
		if self.verbose:
			print(f"Select {num_colors} locations")
		# displaying the image
		
		while len(self.click_locations) < num_colors:
			cv2.imshow(win_name, img)
			cv2.waitKey(1)

		cv2.destroyWindow(win_name)

		for click in self.click_locations:
			# Get color of pixel selected
			if color_type == "BGR":
				color = self.frame[click[1], click[0]]
				colors.append(color)
			elif color_type == "HSV":
				hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
				color = hsv_frame[click[1], click[0]]
				colors.append(color)
			if self.verbose:
				print("Color: ", color)

		# Reset click locations
		self.click_locations = []
		
		return colors

	def _write_on_im(self, frame, msg="Input Text Here", txt_location=(0,0)):
		font=cv2.FONT_HERSHEY_SIMPLEX
		font_size=1
		font_color=(0,0,255)
		thickness=5

		temp_frame = frame.copy()
		return cv2.putText(temp_frame, msg, txt_location, font, font_size, font_color, thickness)

	def _create_window(self, name):
		cv2.namedWindow(winname=name)
		cv2.setMouseCallback(name, self._click_event)

	def close(self):
		cv2.destroyAllWindows()

def create_mask_bounds(colors, closeness=5, verbose=False):
	""" 
	Creates two lists, upper and lower, that are bounds of the single list passed in
	"""
	# FIXME: This may not not work for colors in the red H value spectrum
	# the color red is in the range of [170, 10] because the H value is a circular spectrum
	upper = []
	lower = []
	for color in colors:
		H = np.arange(0, 179)
		up = np.array([color[0] + closeness, 255, 255])
		low = np.array([color[0] - closeness, 100, 100])
		upper.append(up)
		lower.append(low)

	if verbose:
		print("Upper: ", upper)
		print("Lower: ", lower)
	return upper, lower

# driver function
if __name__=="__main__":

	# reading the image
	img = cv2.imread('figures/i_spy_0.jpg', 1)

	img_info = ImageInfoGetter(img, verbose=True)
	ratio = img_info.get_im_ratio()
	colors = img_info.get_img_colors(num_colors=3, color_type="HSV")
	
	# Get the hsv image
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	# Get the bounds of the colors
	upper, lower = create_mask_bounds(colors, verbose=True)

	# Create the mask for the colors
	masks = []
	for i in range(len(colors)):
		mask = cv2.inRange(hsv_img, lower[i], upper[i])
		masks.append(mask)
	
	# Show the image with all the masks applied
	# FIXME: This is not working
	final_mask = masks[0] + masks[1] + masks[2]

	target = cv2.bitwise_and(img, img, mask=final_mask)

	cv2.imshow('Frame', target)
	cv2.waitKey(0)
	cv2.destroyAllWindows()