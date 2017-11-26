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
        min_speed = 1

        params = [wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle]
        self.yaw_controller = YawController(*params)
        self.lowpass = LowPassFilter(3., 1.)
        self.pid = PID(kp=1.4, ki=0.001, kd=0., mn=self.decel_limit, mx=self.accel_limit)
        self.last_time = rospy.get_time()
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

        steer = self.yaw_controller.get_steering(linear_setpoint, angular_setpoint, linear_current)

        # set throttle and brake default to 0.
        # they should never both be > 0. (gas and brake at same time)
        throttle = 0.
        brake = 0.

        # if setpoint velocity less than threshold do max brake (fixes low throttle crawl at light)
        if linear_setpoint < 0.11:
            steer = 0. # yaw controller has trouble at low speed
            brake = abs(self.decel_limit) * self.brake_torque
        else:
            # speeding up - set velocity to throttle 
            if velocity > 0.:
                throttle = velocity
            # slowing down - check deadband limit before setting brake
            else:
                velocity = abs(velocity)

                # brake if outside deadband
                if velocity > self.brake_deadband:
                    brake = velocity * self.brake_torque

        

        print("Throttle: {}  Brake: {}  Steering: {} Linear curr: {}".format(throttle, brake, steer, linear_current))

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
