#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
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

LOOKAHEAD_WPS = 100
STOP_SHIFT = 20

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.current_waypoint_id = None
        self.track_waypoints = None
        self.current_pose = None
        self.current_velocity = None
        self.stop_waypoint_id = -1

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.update()
        rospy.spin()

    # publish next N waypoints to /final_waypoints interval rate
    def update(self):
        #If rate smaller the car stops too late
        rate = rospy.Rate(50)

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
        waypoints = self.track_waypoints
        nearest_waypoint = [0, 100000] # index, ceiling for min distance
        car_pos = self.current_pose.pose.position

        for i in range(len(waypoints)):
            distance = self.euclid_distance(car_pos, waypoints[i].pose.pose.position)
            if(distance < nearest_waypoint[1]):
              nearest_waypoint = [i, distance]
        #TODO This checks only the distance not if the waypoint is behind or in front.
        #TODO It needs to be corrected using the orientation as well
        return nearest_waypoint[0]

    def publish_next_waypoints(self):

        waypoints = Lane()
        waypoints.header.stamp = rospy.Time(0)
        waypoints.header.frame_id = self.current_pose.header.frame_id
        waypoints.waypoints = copy.deepcopy(self.track_waypoints[self.current_waypoint_id: self.current_waypoint_id + LOOKAHEAD_WPS])

        waypoints.waypoints = self.stopping(waypoints.waypoints)
        self.final_waypoints_pub.publish(waypoints)

    def pose_cb(self, msg):
        self.current_pose = msg

    def velocity_cb(self, msg):
        self.current_velocity = msg

    def waypoints_cb(self, waypoints):
        self.track_waypoints = waypoints.waypoints
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.stop_waypoint_id = msg.data

    def stopping(self, waypoints):

        waypoints_to_tl = self.stop_waypoint_id - self.current_waypoint_id
        print waypoints_to_tl, self.stop_waypoint_id, self.current_waypoint_id
        #TODO Stopping decision depends at the moment only on LOOKAHEAD_WPS, we should make it distance/speed dependent
        if(self.stop_waypoint_id >= 0) and ( 0 <= waypoints_to_tl < LOOKAHEAD_WPS):
            print "Breaking"
            distance_to_stop_point = self.wp_distance(self.current_waypoint_id,
                                                   self.stop_waypoint_id - STOP_SHIFT)
            for i in range(waypoints_to_tl - STOP_SHIFT, LOOKAHEAD_WPS):
                waypoints[i].twist.twist.linear.x = 0.0
            for i in range(waypoints_to_tl- STOP_SHIFT):
                current_velocity = math.sqrt(self.current_velocity.twist.linear.x ** 2 +
                                             self.current_velocity.twist.linear.y ** 2)
                #waypoint_velocity = current_velocity*math.sqrt(self.wp_distance(self.current_waypoint_id + i,
                #                                               self.stop_waypoint_id - STOP_SHIFT)/distance_to_stop_point)
                waypoint_velocity = math.sqrt(current_velocity**2 -8.0*self.wp_distance(self.current_waypoint_id + i,
                                                                                        self.current_waypoint_id - STOP_SHIFT))
                if waypoint_velocity < 2.5:
                    waypoint_velocity = 0.0
                waypoints[i].twist.twist.linear.x = waypoint_velocity
        elif self.stop_waypoint_id == -2:
            #TODO CASE OF FAILED DETECTION
            return waypoints

        return waypoints

    def wp_distance(self, wp1, wp2):
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
