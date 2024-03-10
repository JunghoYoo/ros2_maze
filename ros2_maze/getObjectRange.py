# 2024 Mar 10th
# ROS2 maze
# Written by Jung Ho(Julian) Yoo


import sys
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from rclpy.qos import qos_profile_sensor_data
from ros2_maze_custommsg.msg import LidarData


class GetObjectRange(Node):
	def __init__(self):		
		# Creates the node.
		super().__init__('get_object_range')

		self._scan_subscriber = self.create_subscription(
			LaserScan,
			'/scan',
			self._scan_callback,
			qos_profile=qos_profile_sensor_data
		)

		self._scan_subscriber # Prevents unused variable warning.
		self._objectrange_publisher = self.create_publisher(LidarData, 'object_range', 10)
		
		
	def _scan_callback(self, msg):
		self.header = msg.header

		self.angle_min = msg.angle_min
		self.angle_max = msg.angle_max
		self.angle_increment = msg.angle_increment
		self.angles = self.linspace_step(self.angle_min, self.angle_max, step=self.angle_increment)
		self.time_increment = msg.time_increment
		self.scan_time = msg.scan_time
		self.range_min = msg.range_min
		self.range_max = msg.range_max
		self.ranges = msg.ranges
		#self.ranges = self.replace_nan_with_nn(self.ranges) # Warning!!!!! unreliable NaN will be
		self.ranges = self.minus1padding_outliners(self.range_min, self.range_max, self.ranges) # Nan will be replaced with -1
		self.intensities = msg.intensities
		
		# publish
		obj_angle = Float64MultiArray()
		obj_range = Float64MultiArray()
		for i in self.angles:
			obj_angle.data.append(i)
		for i in self.ranges:
			obj_range.data.append(i)

		lidar = LidarData()
		lidar.angles = obj_angle
		lidar.ranges = obj_range
		self._objectrange_publisher.publish(lidar)


	def find_closest(self, array, value):
		idx = (np.abs(array - value)).argmin()
		return idx, array[idx]


	def linspace_step(self, start, stop, step=1.):
		return np.linspace(start, stop, int((stop - start) / step + 1))


	def wrapradian(self, rad):
		# [0 2*pi)
		while rad < 0.0:
			rad += 2*np.pi

		while rad >= 2*np.pi:
				rad -= 2*np.pi
		
		return rad


	def get4waydistances(self, lidar_ranges, lidar_angles):
		# very first elements of lidar data correspond to right side of the image (i.e., higher y values)
		DISTANCE_THRESHOLD = 0.4
		angular_segmentstep = 10 # degree

		angular_segmentinit_radian = 2 * np.pi - np.deg2rad(angular_segmentstep) / 2
		angular_mindistance_segments = []#np.zeros((360//angular_segmentstep, 2))

		lidar_ranges = np.flip(lidar_ranges)
		for angular_segment in np.arange(0, 360, angular_segmentstep): # CW(RIGHT)
			angular_segment_start = self.wrapradian(angular_segmentinit_radian + np.deg2rad(angular_segment))
			angular_segment_end = self.wrapradian(angular_segment_start + np.deg2rad(angular_segmentstep))

			angular_segment_start_index, _ = self.find_closest(lidar_angles, angular_segment_start)
			angular_segment_end_index, _ = self.find_closest(lidar_angles, angular_segment_end)
			if angular_segment_start_index > angular_segment_end_index:
				#lidar_segment_angles = np.concatenate([lidar_angles[angular_segment_start_index:], lidar_angles[:angular_segment_end_index]])    
				lidar_segment_distances = np.concatenate([lidar_ranges[angular_segment_start_index:], lidar_ranges[:angular_segment_end_index]]) 
			else:
				#lidar_segment_angles = lidar_angles[angular_segment_start_index:angular_segment_end_index]
				lidar_segment_distances = lidar_ranges[angular_segment_start_index:angular_segment_end_index]
			
			if lidar_segment_distances.size > 0:
				min_dist = np.min(lidar_segment_distances)
			else: # missing value - too close
				min_dist = 0.0

			print(min_dist < DISTANCE_THRESHOLD, 'angular_segment: ', angular_segment, 'min lidar_segment_distances: ', min_dist)#, angular_segment_start_index, angular_segment_end_index)
			# get minimum distance for each segment
			angular_mindistance_segments.append(min_dist)

		return angular_mindistance_segments


	def get_mindistance_segments(self, lidar_ranges, lidar_angles):
		# very first elements of lidar data correspond to right side of the image (i.e., higher y values)
		DISTANCE_THRESHOLD = 0.4
		angular_segmentstep = 10 # degree

		angular_segmentinit_radian = 2 * np.pi - np.deg2rad(angular_segmentstep) / 2
		angular_mindistance_segments = []#np.zeros((360//angular_segmentstep, 2))

		lidar_ranges = np.flip(lidar_ranges)
		for angular_segment in np.arange(0, 360, angular_segmentstep): # CW(RIGHT)
			angular_segment_start = self.wrapradian(angular_segmentinit_radian + np.deg2rad(angular_segment))
			angular_segment_end = self.wrapradian(angular_segment_start + np.deg2rad(angular_segmentstep))

			angular_segment_start_index, _ = self.find_closest(lidar_angles, angular_segment_start)
			angular_segment_end_index, _ = self.find_closest(lidar_angles, angular_segment_end)
			if angular_segment_start_index > angular_segment_end_index:
				lidar_segment_distances = np.concatenate([lidar_ranges[angular_segment_start_index:], lidar_ranges[:angular_segment_end_index]]) 
			else:
				lidar_segment_distances = lidar_ranges[angular_segment_start_index:angular_segment_end_index]
			
			if lidar_segment_distances.size > 0:
				min_dist = np.min(lidar_segment_distances)
			else: # missing value - too close
				min_dist = 0.0

			print(min_dist < DISTANCE_THRESHOLD, 'angular_segment: ', angular_segment, 'min lidar_segment_distances: ', min_dist)#, angular_segment_start_index, angular_segment_end_index)
			# get minimum distance for each segment
			angular_mindistance_segments.append(min_dist)

		return angular_mindistance_segments


	def get_object_distance(self, object_fov, fov_angles, fov_distances):
		idx, _ = self.find_closest(fov_angles, object_fov)
		return fov_distances[idx]


	def minus1padding_outliners(self, min, max, array):
		array = np.array(array)
		newarray = np.zeros_like(array)

		for i, data in enumerate(array):
			if data >= min and data <= max:
				newarray[i] = data
			else:
				newarray[i] = -1.0 # unreliable data

		return np.array(newarray)


	def replace_nan_with_nn(array):
		array = np.array(array)
		mask = np.isnan(array)
		array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
		return array


def main(argv=sys.argv):
    rclpy.init(args=argv)
    node = GetObjectRange()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
	main()
