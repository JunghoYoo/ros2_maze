# 2024 Mar 10th
# ROS2 maze
# Written by Jung Ho(Julian) Yoo


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage # real HW
from sensor_msgs.msg import Image # Gazebo
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge
from ros2_maze_custommsg.srv import ImagePred

import os
import pickle

# KNN
k = 7 # odd number

# Image Resize for KNN
WRESIZE = 33
HRESIZE = 25

CROPMARGIN = 10

# Apply orange/green/blue mask (separately, 1 ch) from HSV image
def ApplyColorMask(image_HSV):  
    # Max value for Y is 180
    # https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    S_MIN = 150
    S_MAX = 255
    V_MIN = 125
    V_MAX = 200

    lower_orange = np.array([170, S_MIN, V_MIN], dtype = "uint8") 
    upper_orange = np.array([180, S_MAX, V_MAX], dtype = "uint8")
    mask = cv2.inRange(image_HSV, lower_orange, upper_orange)
    orange1 = cv2.bitwise_and(image_HSV, image_HSV, mask = mask)

    S_MIN = 150
    S_MAX = 255
    V_MIN = 0
    V_MAX = 200
    
    lower_orange = np.array([0, S_MIN, V_MIN], dtype = "uint8") 
    upper_orange = np.array([10, S_MAX, V_MAX], dtype = "uint8")  
    mask = cv2.inRange(image_HSV, lower_orange, upper_orange)
    orange2 = cv2.bitwise_and(image_HSV, image_HSV, mask = mask)
    orange = (orange1 + orange2) / 2
    orange = np.sum(orange, axis=2) / 3
    orange = orange.astype(np.uint8)
    
    S_MIN = 75
    S_MAX = 255
    V_MIN = 0
    V_MAX = 150
        
    lower_blue = np.array([100, S_MIN, V_MIN], dtype = "uint8")
    upper_blue = np.array([120, S_MAX, V_MAX], dtype = "uint8")
    mask = cv2.inRange(image_HSV, lower_blue, upper_blue)
    blue = cv2.bitwise_and(image_HSV, image_HSV, mask = mask)
    blue = np.sum(blue, axis=2) / 3
    blue = blue.astype(np.uint8)
    
    S_MIN = 100
    S_MAX = 255
    V_MIN = 0
    V_MAX = 150
    
    lower_green = np.array([50, S_MIN, V_MIN], dtype = "uint8")
    upper_green = np.array([80, S_MAX, V_MAX], dtype = "uint8")
    mask = cv2.inRange(image_HSV, lower_green, upper_green)
    green = cv2.bitwise_and(image_HSV, image_HSV, mask = mask)
    green = np.sum(green, axis=2) / 3
    green = green.astype(np.uint8)
    
    # horizontal upper will be masked as zero
    UPPERCUT = 50
    orange[:UPPERCUT,:] = 0
    blue[:UPPERCUT,:] = 0
    green[:UPPERCUT,:] = 0
    
    return orange, blue, green 


def GetBoxForMaxContour(colormask):
    # non zero will be 1
    ret, thresh = cv2.threshold(colormask, 1, 255, cv2.THRESH_BINARY)
    # Find Outer(cv2.RETR_EXTERNAL) contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # number of found countours
    #if __debug__:   
    #    print("# of counturs: ", len(contours))
    
    maxarea = 0.0
    box = np.zeros(4)
    
    # Draw contours
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    index = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if maxarea < area:
            maxarea = area
            index = i
             
        if __debug__:     
            color = (255,255,255)
            
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    
    if len(contours) > 0:
        lefttop = np.min(contours[index], axis=0)
        rightbottom = np.max(contours[index], axis=0)
        x = lefttop[0,0]
        y = lefttop[0,1]
        w = rightbottom[0,0] - x
        h = rightbottom[0,1] - y
        box = np.array([x, y, w, h], dtype=np.int32)

    return maxarea, drawing, box


def CropnResize(image, box, size, margin = CROPMARGIN):
    # crop only max area & resize
    y1 = box[1] - margin; y2 = box[1] + box[3] + margin
    x1 = box[0] - margin; x2 = box[0] + box[2] + margin
    y1 = max(0, y1); x1 = max(0, x1)
    y2 = min(image.shape[0]-1, y2); x2 = min(image.shape[1]-1, x2)
    
    croppedimage = image[y1:y2,x1:x2,:]
    resizeimage = cv2.resize(croppedimage, size)
    
    return resizeimage

            
def load_model():
    filename = 'src/team23_maze_final/team23_maze_final/mydt_model.pkt'
    dt = pickle.load(open(os.path.join(os.getcwd(), filename), 'rb'))
    filename = 'src/team23_maze_final/team23_maze_final/myknn_model.xml'
    knn = cv2.ml.KNearest.load(os.path.join(os.getcwd(), filename))
    return dt, knn


def predict(image_BGR, dt, knn):
    test_images = []
    test_boxnareas = []
    neighbours = None
    dist = None
    
    # preprocessing with HSV
    image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    orange, blue, green = ApplyColorMask(image_HSV)
    areaorange, drawingorange, boxorange = GetBoxForMaxContour(orange)
    areablue, drawingblue, boxblue = GetBoxForMaxContour(blue)
    areagreen, drawinggreen, boxgreen = GetBoxForMaxContour(green)

    # select color which has max area of contour
    areas = np.array([areaorange, areablue, areagreen])
    box = np.array([boxorange, boxblue, boxgreen], dtype=np.int32)
    index = np.argmax(areas)

    size = (WRESIZE,HRESIZE)
    
    test_boxnareas = np.array([box[index][0], box[index][1], box[index][2], box[index][3], areas[index]])
    test_boxnareas = np.expand_dims(test_boxnareas, axis=0)
    
    # crop only max area & resize 
    resizeimage_HSV = CropnResize(image_HSV, box[index], size)
    # only for S channel will be used for later
    resizeimage_S = resizeimage_HSV[:,:,1]
        
    test_images = np.array(resizeimage_S)
    test_images = test_images.flatten().reshape(-1, HRESIZE*WRESIZE)
    test_images = test_images.astype(np.float32)
    
    # decision tree
    dt_pred = dt.predict(test_boxnareas,)
    
    if dt_pred > 0: # knn
        ret, results, neighbours, dist = knn.findNearest(test_images, k)
        ret = np.int32(ret)
    else: # class 0 is decided by decision tree
        ret = 0
    
    return ret, resizeimage_S, neighbours, dist


class imageClassifier(Node):

	def __init__(self):		
		# Creates the node.
		super().__init__('imageClassifier')

		# Set Parameters
		self.declare_parameter('show_image_bool', True) # False when executing on robot
		self.declare_parameter('window_name', "Raw Image")

		#Determine Window Showing Based on Input
		self._display_image = bool(self.get_parameter('show_image_bool').value)

		# Declare some variables
		self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
		
		#Only create image frames if we are not running headless (_display_image sets this)
		if(self._display_image):
		# Set Up Image Viewing
			cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
			cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
		
		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

		#Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
		self._video_subscriber = self.create_subscription(
				#CompressedImage # real hw
				#'/image_raw/compressed', # real hw
				Image,
                '/camera/image_raw', # gazebo simulation
				self._imageshow_callback,
				image_qos_profile)
		self._video_subscriber # Prevents unused variable warning.

		#Image Classification Server
		self._image_srv = self.create_service(ImagePred, 'image_clasification', self._imageclassification_callback)
		self._image_srv # Prevents unused variable warning.
		
		#Load trained model
		self._mydt, self._myknn = load_model()
		print("load model ok")
		
        # image subscriber
		self._imgBGR = None


	def _imageclassification_callback(self, request, response):
		# Only when request service, image classification is executed.
		pred, resizeimage_S, neighbours, dist = predict(self._imgBGR, self._mydt, self._myknn)
		SIGN = { 0 : "EMPTYWALL", 1 : "LEFTSIGN", 2 : "RIGHTSIGN", 3 : "DNESIGN", 4 : "STOPSIGN", 5 : "GOALSIGN" }
		print("Classified : ", SIGN[pred])
		response.pred = int(pred)

		return response


	def _imageshow_callback(self, Image):	# CompressImage for real HW
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        #self._imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8") # real HW
		self._imgBGR = CvBridge().imgmsg_to_cv2(Image, "bgr8") # gazebo
  		
		if(self._display_image):
			# Display the image in a window
			self.show_image(self._imgBGR)


	def show_image(self, img):
		cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
		self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.


	def get_user_input(self):
		return self._user_input


def main():
	rclpy.init() #init routine needed for ROS2.
	classifier = imageClassifier() #Create class object to be used.

	while rclpy.ok():
		rclpy.spin_once(classifier) # Trigger callback processing.
		if(classifier._display_image):	
			if classifier.get_user_input() == ord('q'):
				cv2.destroyAllWindows()
				break

	#Clean up and shutdown.
	classifier.destroy_node()  
	rclpy.shutdown()


if __name__ == '__main__':
	main()
