# 2024 Mar 10th
# ROS2 maze
# Written by Jung Ho(Julian) Yoo


import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from turtlebot3_msgs.srv._sound import Sound
from geometry_msgs.msg import Quaternion, Point
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
from action_msgs.msg._goal_status_array import GoalStatusArray
from action_msgs.msg._goal_info import GoalInfo
from action_msgs.msg._goal_status import GoalStatus
import tf_transformations
import tf2_ros
from ros2_maze_custommsg.msg import LidarData
from ros2_maze_custommsg.srv import ImagePred

import numpy as np
import math


STATE_DICT = {
    0: 'Start',
    1: 'GoToSign',
    2: 'WaitBeforeImageClassification',
    3: 'DecideWhereToGo',
    4: 'GobackToPreviousSign',
    9: 'Goal',
    10: 'End'
}

WAITING_SECS = 1
DISTANCEFROMSIGN = 0.5
IMAGECLASSIFICATION_RETRY = 3
IMAGECLASSIFICATION_WAITING_SECS = 1
PUBLISHGOAL_WAITING_SECS = 1

# label for sign prediction
EMPTYWALL = 0
LEFTSIGN = 1
RIGHTSIGN = 2
DNESIGN = 3
STOPSIGN = 4
GOALSIGN = 5


# For a client in service, seperate class is required and then make request in the other class.
class imageClassifyClient(Node):
    def __init__(self):
        super().__init__('imageService')

        self.image_client = self.create_client(ImagePred, 'image_clasification')
        while not self.image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.image_request = ImagePred.Request()


    def reqImagePred(self):
        self.future = self.image_client.call_async(self.image_request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


class goToGoal(Node):
    def __init__(self):
        super().__init__('goTogoal')

        # sound disable under gazebo env
        # self.SOUND_VALUES = {
        #     'OFF': 0,
        #     'ON': 1,
        #     'LOW_BATTERY': 2,
        #     'ERROR': 3,
        #     'BUTTON1': 4,
        #     'BUTTON2': 5,
        # }

        self.state = 0

        self.object_range_subscription = self.create_subscription(
            LidarData,
            'object_range',
            self._object_range_callback,
            10)
    
        self.TIMER_PERIOD = 0.1  # Control or state machine update time (Digital Control) 
        self.tick_timeout = 0.0
        self.timer = self.create_timer(self.TIMER_PERIOD, self.timer_callback)

        # self.sound_cli = self.create_client(Sound, '/sound')
        # while not self.sound_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('sound service not available, waiting again...')
        # self.sound_req = Sound.Request()
    
        self.image_client = self.create_client(ImagePred, 'image_clasification')
        while not self.image_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('image classification service noDecideWhereToGot available, waiting again...')
        self.image_req = ImagePred.Request()
    
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publish_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.subscription_goalfeedback = self.create_subscription(
            GoalStatusArray(),
            '/navigate_to_pose/_action/status',
            self._callback_goalstatus,
            10)
        self.goalstatus = GoalStatus.STATUS_EXECUTING
        self._lastangleindex = 0
        self._retry_count = 0
        self.currentidealangle = None # no noise, ideal angle


    def soundservice_request(self, value):
        self.sound_req.value = value
        self.future = self.sound_cli.call_async(self.sound_req)
        #rclpy.spin_until_future_complete(self, self.future) #cannnot check end
        return self.future.result()
    
        
    def _callback_goalstatus(self, msg):
        status_list = msg.status_list
        self.goalstatus = status_list[-1].status # latest
        if status_list[-1].status == GoalStatus.STATUS_SUCCEEDED:# and\
            self.goalstatus = GoalStatus.STATUS_SUCCEEDED
        else:
            self.goalstatus = GoalStatus.STATUS_EXECUTING
    

    def _reachtosign(self):
        if self.goalstatus == GoalStatus.STATUS_SUCCEEDED:
            return True
        else:
            return False
           

    def wrapradian(self, rad):
        # [0 +2pi)
        while rad < 0:
            rad += 2*np.pi

        while rad >= 2*np.pi:
                rad -= 2*np.pi
        
        return rad


    def nearestquaternion(self, rad):
        # [0 2*pi)
        rad = self.wrapradian(rad)
        alternativerad = 2*np.pi - rad
        
        if rad < alternativerad:
            return rad
        else:
            return alternativerad


    def _object_range_callback(self, msg):
        self.obj_angle = []
        for i in range(1, len(msg.angles.data)):
            self.obj_angle.append(msg.angles.data[i])
        self.obj_angle = np.array(self.obj_angle)

        self.obj_range = []
        for i in range(1, len(msg.ranges.data)):
            self.obj_range.append(msg.ranges.data[i])
        self.obj_range = np.array(self.obj_range)


    def timer_callback(self):
        self.state_machine()
        if self.tick_timeout >= 0:
            self.tick_timeout -= self.TIMER_PERIOD


    def _waitsec(self, timeout_sec):
        #self.soundservice_request(self.SOUND_VALUES["ERROR"])
        
        if (self.tick_timeout == 0.0):
            self.tick_timeout = timeout_sec
            return False
        else:
            if self.tick_timeout < 0.0:
                self.tick_timeout = 0.0
                return True
            else:
                return False


    # for client in service, separate class is neeeded!
    # otherwise, it will not check response
    def predictImage(self):
        i = imageClassifyClient()
        response = i.reqImagePred()
        print(response.pred)
        return response.pred


    def _decide_dest(self, isStart = False):
        if isStart: 
            # go forward
            if self._publish_dest(0):
                pred = 0
                self._lastangleindex = pred
            else:
                return -1
        else:
            pred = self.predictImage()
            print(pred)
            
            # 0 empty wall, 1 left, 2 right, 3 do not enter, 4 stop, 5 goal
            if pred in [LEFTSIGN, RIGHTSIGN, DNESIGN]:
                fourways = [0.0, (np.pi/2), (np.pi), (3*np.pi/2)] # not wall, left, right, uturn (CCW)

                if self._publish_dest(fourways[pred]):
                    self._lastangleindex = pred
                    return pred
                else:
                    print("Too close from the destination")
                    return -1
            elif pred is STOPSIGN:
                self._publish_dest()
                self._lastangleindex = pred
            elif pred in [EMPTYWALL]:
                return pred
            elif pred in [GOALSIGN]:
                return pred

        return pred


    def _gobacktoprev(self):
        inversefourways = [(np.pi), (3*np.pi/2), (np.pi/2), 0.0] # not wall, left, right, uturn
        if self._publish_dest(inversefourways[self._lastangleindex]):
            self._lastangleindex = 0
            return 0
        else:
            return -1


    def _get_currentpos(self):
        try:
            # Get current pose of the robot by transforming coordinations from map to robot's base_link
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            #self.get_logger().error('Failed to get current robot pose: ' + str(e))
            return None

			
    def find_closest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]


    def _publish_dest(self, decided_angle):
        decided_angle = self.wrapradian(decided_angle) # angle -> rad
        
        currentpos = self._get_currentpos()
		# Calculate the index in lidar readings corresponding to the desired direction
        if currentpos is None:
            print("tf error")
            return False
        
        current_angle = tf_transformations.euler_from_quaternion([
			currentpos.rotation.x,
			currentpos.rotation.y,
			currentpos.rotation.z,
			currentpos.rotation.w])[2]

        # heading angle (ideal without noise)
        if self.currentidealangle is None:
            self.currentidealangle = current_angle

        dest_angle = self.currentidealangle + decided_angle
        dest_angle = self.wrapradian(dest_angle) # global frame
        self.currentidealangle = dest_angle

        # Decide the distance to the angle by lidar data
        index, _ = self.find_closest(self.obj_angle, decided_angle)  # local frame
        dest_dist = self.obj_range[index] - DISTANCEFROMSIGN # distance from wall -> DISTANCEFROMSIGN
        print("angle: ", dest_angle, "Lidar index: ", index, "/", len(self.obj_range), " distance: ", dest_dist)   
        if dest_dist <= 0.0: # Lidar NaN or wrong image classificagtion
            print("Too close destination distance: ", dest_dist, " Destination angle: ", dest_angle)
            dest_dist = self.obj_range[index] 

        # Calculate new coordination for the destination
        dest_position = self.convert2pos(currentpos.translation, dest_angle, dest_dist)
        dest_orientation = self.tr_euler2quaternion(dest_angle)

        # Publish coordination of destination
        dest = PoseStamped()
        dest.header.frame_id = 'map'
        dest.pose.position = Point(**dest_position)
        dest.pose.orientation = Quaternion(**dest_orientation)
        print("Published Destination:", dest, " Distance: ",dest_dist,"current_pos: ",currentpos,"Destination pos: ",dest_position)
        self.publish_goal.publish(dest)

        return True


    def convert2pos(self, currentpos, dest_angle, dest_dist):
        # Calculate new position based on current position, angle, and distance
        new_x = currentpos.x + dest_dist * math.cos(dest_angle)
        new_y = currentpos.y + dest_dist * math.sin(dest_angle)
        return {'x': new_x, 'y': new_y, 'z': 0.0}


    def tr_euler2quaternion(self, angle):
        # Calculate quaternion orientation from an angle
        q = tf_transformations.quaternion_from_euler(0, 0, angle)
        return {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]}


    def state_machine(self):
        print(STATE_DICT[self.state])

        if (STATE_DICT[self.state] == STATE_DICT[0]):
            if self._decide_dest(isStart = True) >= 0:
                self.state += 1
        elif (STATE_DICT[self.state] == STATE_DICT[1]):
            if self._reachtosign():
                self.state += 1
        elif (STATE_DICT[self.state] == STATE_DICT[2]):
            if self._waitsec(IMAGECLASSIFICATION_WAITING_SECS):
                self.state += 1
        elif (STATE_DICT[self.state] == STATE_DICT[3]):
            pred = self._decide_dest()
            self._waitsec(PUBLISHGOAL_WAITING_SECS)

            if pred in [LEFTSIGN, RIGHTSIGN, DNESIGN, STOPSIGN]:
                self.state = 1
            elif pred in [EMPTYWALL]:
                self._retry_count += 1
                if self._retry_count > IMAGECLASSIFICATION_RETRY:
                   self._retry_count = 0
                   self.state = 4 # retry - go back
                else:
                   self.state = 2 # retry - image classification
            elif pred in [GOALSIGN]:
                self.state = 9 # goal
        elif (STATE_DICT[self.state] == STATE_DICT[4]):
            if self._gobacktoprev() >= 0 :
                self.state = 1
            else:
                print("Fail to go back")
                self.state = 10
        elif (STATE_DICT[self.state] == STATE_DICT[9]):
            if self._waitsec(WAITING_SECS):
                print("Arrived at the goal!")
                self.state += 1
        elif (STATE_DICT[self.state] == STATE_DICT[10]):
            pass # fail


def main(args=None):
    rclpy.init(args=args)

    gotoGoal = goToGoal()
    rclpy.spin(gotoGoal)

    gotoGoal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
