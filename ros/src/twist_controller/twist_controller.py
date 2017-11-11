from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        vehicle_mass = kwargs['vehicle_mass']
        fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        
        wheel_radius = kwargs['wheel_radius']
        wheel_base = kwargs['wheel_base']
        steer_ratio = kwargs['steer_ratio']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']
        min_speed = 10

        params = [wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle]
        self.yaw_controller = YawController(*params)
        self.lowpass = LowPassFilter(0.2, 0.1)
        self.pid = PID(kp=0.9, ki=0.0005, kd=0.06, mn=self.decel_limit, mx=self.accel_limit)
        self.last_time = rospy.get_time()
        # https://www.researchgate.net/post/How_can_we_calculate_the_required_torque_to_move_a_massive_object_by_means_of_gear_assembly2
        self.brake_torque = (vehicle_mass + fuel_capacity * GAS_DENSITY) * wheel_radius

    def get_delta(self):
        now = rospy.get_time()
        delta = now - self.last_time if self.last_time else 0.1
        last_time = now
        return delta


    def control(self, *args, **kwargs):
        linear_setpoint = kwargs['linear_setpoint']
        angular_setpoint = kwargs['angular_setpoint']
        linear_current = kwargs['linear_current']

        velocity_error = linear_setpoint - linear_current

        delta = self.get_delta()

        unfiltered = self.pid.step(velocity_error, delta)
        velocity = self.lowpass.filt(unfiltered)



        brake = 0.
        if velocity <= 0.:
            velocity = abs(velocity)
            brake = self.brake_torque * velocity if velocity > self.brake_deadband else 0.
            velocity = 0

        throttle = velocity
        steer = self.yaw_controller.get_steering(linear_setpoint, angular_setpoint, linear_current)

        return throttle, brake, steer
