from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy
import pickle
import copy
import os

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, *args, **kwargs):

        self.controller_log_time_seconds = 5
        self.controller_log_file_base = os.path.join(rospy.get_param('~log_path'), 'controller_id_log_')
        self.controller_logging = rospy.get_param('~logging')

        print (self.controller_log_file_base, self.controller_logging)

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


        self.log = {    'linear_setpoint': [], 
                        'linear_current':  [],
                        'velocity_error':  [],
                        'delta_time':     [],
                        'unfilt_pid_output': [],
                        'lowpass_filt':    [],
                        'timestamp':    [],
                        }

    def get_delta(self):
        now = rospy.get_time()
        delta = now - self.last_time if self.last_time else 0.1
        self.last_time = now
        return delta


    def control(self, *args, **kwargs):
        linear_setpoint = kwargs['linear_setpoint']
        angular_setpoint = kwargs['angular_setpoint']
        linear_current = kwargs['linear_current']

        velocity_error = linear_setpoint - linear_current

        delta = self.get_delta()
        unfiltered = self.pid.step(velocity_error, delta)
        velocity = self.lowpass.filt(unfiltered)
        # print("Linear Setpoint: {}  Linear Current: {}  Velocity Error: {}  Delta: {}  Unfiltered PID Output: {}  Lowpass Filtered: {}"
        #     .format(linear_setpoint, linear_current, velocity_error, delta, unfiltered, velocity))

        brake = 0.
        if velocity <= 0.:
            velocity = abs(velocity)
            brake = self.brake_torque * velocity if velocity > self.brake_deadband else 0.
            velocity = 0

        throttle = velocity
        steer = self.yaw_controller.get_steering(linear_setpoint, angular_setpoint, linear_current)

        print("Throttle: {}  Brake: {}  Steering: {}".format(throttle, brake, steer))

        self.log['linear_setpoint'].append(linear_setpoint)
        self.log['linear_current'].append(linear_current)
        self.log['velocity_error'].append(velocity_error)
        self.log['delta_time'].append(delta)
        self.log['unfilt_pid_output'].append(unfiltered)
        self.log['lowpass_filt'].append(velocity)
        self.log['timestamp'].append(rospy.get_time())
        
        # assumes 50 hz
        if ((len(self.log['timestamp']) > 50 * self.controller_log_time_seconds)) and self.controller_logging:
            log_to_save = copy.deepcopy(self.log)

            log_iteration = 0
            log_file = self.controller_log_file_base + str(log_iteration) + '.pkl'
            while(os.path.isfile(log_file)):
                log_iteration += 1
                log_file = self.controller_log_file_base + str(log_iteration) + '.pkl'
            
            if(log_iteration<100): # don't save a million files
                with open(log_file, 'wb') as file:
                    pickle.dump(log_to_save, file)

            self.log['linear_setpoint'] = []
            self.log['linear_current'] = []
            self.log['velocity_error'] = []
            self.log['delta_time'] = []
            self.log['unfilt_pid_output'] = []
            self.log['lowpass_filt'] = []
            self.log['timestamp'] = []


        return throttle, brake, steer