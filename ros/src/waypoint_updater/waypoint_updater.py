#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf as ros_tf
import math
import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 80
STOP_SHIFT_m = 5
#TODO Generalize for Carla
DEC_MAX = -4

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.current_waypoint_id = None
        self.track_waypoints = None
        self.current_pose = None
        self.current_yaw = None
        self.current_velocity = None
        self.tl_waypoint_id = -1

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.update()
        rospy.spin()

    # publish next N waypoints to /final_waypoints interval rate
    def update(self):
        #If rate smaller the car stops too late
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
          if self.track_waypoints and self.current_pose:
            self.current_waypoint_id = self.find_nearest_waypoint()
            self.publish_next_waypoints()
          rate.sleep()

    # calculate the euclidian distance between our car and a waypoint
    #TODO This is a static function
    def euclid_distance(self, car_pos, wpt_pos):
        a = np.array((car_pos.x, car_pos.y))
        b = np.array((wpt_pos.x, wpt_pos.y))
        return np.linalg.norm(a-b)

    # find index of nearest waypoint in self.track_waypoints
    def find_nearest_waypoint(self):
        nearest_waypoint = [0, 100000] # index, ceiling for min distance
        car_pos = self.current_pose.pose.position

        for i in range(len(self.track_waypoints)):
            wp_pos = self.track_waypoints[i].pose.pose.position
            distance = self.euclid_distance(car_pos, wp_pos)
            angle = math.atan2(car_pos.y - wp_pos.y, car_pos.x - wp_pos.x)
            if(distance < nearest_waypoint[1]) and (abs(angle - self.current_yaw) < math.pi / 4.0):
                nearest_waypoint = [i, distance]
        return nearest_waypoint[0]

    def publish_next_waypoints(self):

        waypoints = Lane()
        waypoints.header.stamp = rospy.Time(0)
        waypoints.header.frame_id = self.current_pose.header.frame_id
        waypoints.waypoints = copy.deepcopy(self.track_waypoints[self.current_waypoint_id: self.current_waypoint_id
                                                                                           + LOOKAHEAD_WPS])
        waypoints.waypoints = self.stopping(waypoints.waypoints)
        self.final_waypoints_pub.publish(waypoints)

    def pose_cb(self, msg):
        self.current_pose = msg
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.current_yaw = ros_tf.transformations.euler_from_quaternion(quaternion)[2]

    def velocity_cb(self, msg):
        self.current_velocity = msg

    def waypoints_cb(self, waypoints):
        self.track_waypoints = waypoints.waypoints
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.tl_waypoint_id = msg.data

    def stopping(self, waypoints):

        waypoints_to_tl = self.tl_waypoint_id - self.current_waypoint_id
        if self.tl_waypoint_id >= 0:
            distance_to_tl = self.wp_distance(self.current_waypoint_id, self.tl_waypoint_id)
        else:
            distance_to_tl = 1000000
        current_speed = math.sqrt(self.current_velocity.twist.linear.x ** 2 +
                                     self.current_velocity.twist.linear.y ** 2)
        try:
            acc_needed = current_speed**2/(2*distance_to_tl)
        except ZeroDivisionError:
            acc_needed = 10000.0


        #TODO Compose to one if, but at the moment more clear like this
        
        # If the average acceleration needed is greater than max acceleration (used when we see red and we are already
        # too close to attempt a stop
        #if acc_needed > abs(DEC_MAX):
        #    print "Too late for stopping....brakes not powerful enough..."
        #    pass
        
        # A green light or no traffic light is observed
        if self.tl_waypoint_id < 0:
            print "No red light is observed..."
            pass

        # This should not happen apart from asyncr situation. The traffic light needs to be in front of the car
        elif waypoints_to_tl < 0:
            return  waypoints
        
        # If we are still too far from the traffic light there is not reason to start braking already
        elif distance_to_tl > 50:
            print "Still too early to start braking...chill yo.."
            pass
        
        # Handle cases of traffic_waypoint = -2 (failed detection)
        elif self.tl_waypoint_id == -2:
            pass
        
        # All other cases that will impose a braking
        else:
            # Finding the stopping waypoint in terms of distance to the traffic light waypoint
            stop_waypoint = self.tl_waypoint_id
            for i in range(waypoints_to_tl):
                d = self.wp_distance(self.tl_waypoint_id - i, self.tl_waypoint_id)
                #TODO Working on a better condition
                if abs(d - STOP_SHIFT_m) < 1.0:
                    stop_waypoint -= i
                    break

            print "Breaking, stop waypoint index : ", stop_waypoint
            print "Current speed: ", current_speed

            dist_to_stop_wp = self.wp_distance(self.current_waypoint_id, stop_waypoint)
            print "Distance to stop_waypoint: ", dist_to_stop_wp

            wp_to_stop = stop_waypoint - self.current_waypoint_id
            print "Waypoints until stops: ", wp_to_stop


            for i in range(wp_to_stop, LOOKAHEAD_WPS):
                waypoints[i].twist.twist.linear.x = 0.0
            for i in range(wp_to_stop):

                deceleration = -current_speed**2/(2*self.wp_distance(self.current_waypoint_id, stop_waypoint))

                deceleration = -4.0
                try:
                    waypoint_velocity = math.sqrt(current_speed**2 +
                                              2*deceleration*self.wp_distance(self.current_waypoint_id,
                                                                          self.current_waypoint_id + i))
                except:
                    waypoint_velocity = 0.0
                #waypoint_velocity = current_speed*math.sqrt(self.wp_distance(self.current_waypoint_id + i,
                #                                                             stop_waypoint)/dist_to_stop_wp)
                if current_speed < 0.01:
                    waypoint_velocity = 0.0
                if waypoint_velocity < 2.5:
                    waypoint_velocity = 0.0
                waypoints[i].twist.twist.linear.x = waypoint_velocity
        self.debug_waypoint_velocity(waypoints)
        return waypoints

    def debug_waypoint_velocity(self, wps):
        speeds = [str(wps[i].twist.twist.linear.x)[:4] for i in range(len(wps))]
        print speeds
        print


    def wp_distance(self, wp1, wp2):
        #TODO Attention when the track closes
        assert wp1 <= wp2
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for i in range(wp2 - wp1):
            dist += dl(self.track_waypoints[wp1 + i].pose.pose.position,
                       self.track_waypoints[wp1 + i + 1].pose.pose.position)
        return dist



    def wp_distance2(self, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(self.track_waypoints[wp1].pose.pose.position,
                       self.track_waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
