#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.current_waypoints = None
        self.current_pose = None

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.current_waypoint_pub = rospy.Publisher('current_waypoint', Int32, queue_size=1)

        self.update()
        rospy.spin()

    # publish next N waypoints to /final_waypoints interval rate
    def update(self):
        interval = rospy.Rate(1)

        while not rospy.is_shutdown():
          if(self.current_waypoints and self.current_pose):
            nearest_waypoint = self.find_nearest_waypoint()

            self.publish_next_waypoints(nearest_waypoint)

          interval.sleep()

    # calculate the euclidian distance between our car and a waypoint
    def calculate_distance(self, car_pos, wpt_pos):
        a = np.array((car_pos.x, car_pos.y, car_pos.z))
        b = np.array((wpt_pos.x, wpt_pos.y, wpt_pos.z))

        distance = np.linalg.norm(a-b)

        return distance

    # find index of nearest waypoint in self.current_waypoints
    def find_nearest_waypoint(self):
        waypoints = self.current_waypoints
        nearest_waypoint = [0, 100000] # index, ceiling for min distance
        car_pos = self.current_pose.pose.position

        # loop through waypoints and find min distance
        for i in range(len(waypoints)):
            distance = self.calculate_distance(car_pos, waypoints[i].pose.pose.position)
            if(distance < nearest_waypoint[1]):
              nearest_waypoint = [i, distance]

        return nearest_waypoint[0]

    # publish a list of next n waypoints to /final_waypoints
    def publish_next_waypoints(self, start_index):
        waypoints = Lane()

        waypoints.header.stamp = rospy.Time(0)
        waypoints.header.frame_id = self.current_pose.header.frame_id

        waypoints.waypoints = self.current_waypoints[start_index:start_index + LOOKAHEAD_WPS]

        # publish index of closest waypoint
        self.current_waypoint_pub(start_index)

        # publish next N waypoints
        self.final_waypoints_pub.publish(waypoints)

    def pose_cb(self, msg):
        self.current_pose = msg

    def waypoints_cb(self, waypoints):
        self.current_waypoints = waypoints.waypoints

        # we only need the message once, unsubscribe as soon as we got the message
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
