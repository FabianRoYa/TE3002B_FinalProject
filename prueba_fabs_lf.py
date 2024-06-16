import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class VisionClass(Node):
    def __init__(self):
        super().__init__('vision')
        self.detected_objects = {}  # Dictionary to store detected objects
        self.last = 0  # Last time an object was detected
        self.type = ''  # Type of detected object
        self.timer_period = 0.05  # Timer period in seconds
        self.w_img = 320  # Image width
        self.w_t = int(self.w_img / 2)  # Half the image width
        self.area_min_light = 200  # Minimum area for light detection
        self.area_min_line = 2000  # Minimum area for line detection
        self.angle = 0.0  # Calculated angle of the detected line
        self.error_line = 0  # Error in the line position
        self.x_line = 0  # X position of the detected line
        self.min_v = 0.01  # Minimum robot speed
        self.max_v = 0.15  # Maximum robot speed
        self.max_w = 0.55  # Maximum robot angular speed
        self.kp = 0.005  # Proportional constant for line control
        self.kp_curve = 2.0  # Proportional constant for curve control
        self.tolerance = 10  # Tolerance for line error
        self.slow_rate = 0.3  # Deceleration rate
        self.f_v = 0.0  # Linear speed calculated by the fuzzy controller
        self.f_w = 0.0  # Angular speed calculated by the fuzzy controller
        self.img = []  # Image received from the camera
        self.img1 = []  # Processed sub-image 1
        self.img2 = []  # Processed sub-image 2
        self.robot_state = "green"  # Robot state (used for traffic lights)
        self.detect_line = False  # Indicator of line detection
        self.image_received = False  # Indicator of image reception
        self.move = False  # Indicator of robot movement
        self.controller = None  # Fuzzy controller
        self.bridge = CvBridge()  # Initialize CvBridge to convert images
        self.robot_vel = Twist()  # Initialize the robot velocity message
        self.inf = String()  # Information message
        self.last_signal = None  # Last detected signal
        self.semaforo_state = 'green'  # Traffic light state
        self.val_semaforo = 1  # Traffic light value to modify speed
        self.previous_val_semaforo = 1  # Previous traffic light value
        self.last_stop_robot = 0  # Last stop robot state
        
        # Subscriptions
        self.sub_image = self.create_subscription(Image, 'video_source/raw', self.camera_callback, 10)
        self.subscription = self.create_subscription(String, '/detected_object', self.detected_object_callback, 10)
        
        # Publishers
        self.pub_image1 = self.create_publisher(Image, 'image_process1', 10)
        self.pub_image2 = self.create_publisher(Image, 'image_process2', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub = self.create_publisher(String, 'inf', 10)
        
        self.timer = self.create_timer(self.timer_period, self.timer_callback)  # Create a timer
        self.controller_make()  # Create the fuzzy controller
        self.get_logger().info('Vision node initialized')  # Show start message in logger
        self.og_vel = Twist()  # Original robot velocity
        self.stop_robot = 0  # Indicator if the robot should stop
        self.semaforo = 0  # Traffic light state
        self.stop_start_time = None  # Stop start time
        self.original_velocity = Twist()  # Original robot velocity
        self.turn_start_time = None  # Turn start tim
        self.dir = 0
        self.c1 = 0
        self.c2 = 0

    def controller_make(self):
        """
        Function to create the fuzzy control system.
        Defines input and output variables, membership functions, and fuzzy rules.
        """
        angle = ctrl.Antecedent(np.linspace(-70, 70, 180), 'angle')
        x = ctrl.Antecedent(np.linspace(-75, 75, 100), 'x')
        v = ctrl.Consequent(np.linspace(0, 0.15, 80), 'v')
        w = ctrl.Consequent(np.linspace(-0.8, 0.8, 80), 'w')
        angle.automf(5, names=['neg big', 'neg small', 'neutr', 'pos small', 'pos big'])
        x.automf(5, names=['left big', 'left small', 'center', 'right small', 'right big'])
        v.automf(5, names=['very slow', 'slow', 'medium', 'fast', 'very fast'])
        w.automf(5, names=['neg_fast', 'neg_slow', 'neutr', 'pos_slow', 'pos_fast'])
        rules = [
            ctrl.Rule(angle['pos big'], w['pos_fast']),
            ctrl.Rule(angle['pos small'], w['pos_slow']),
            ctrl.Rule(angle['neg small'], w['neg_slow']),
            ctrl.Rule(angle['neg big'], w['neg_fast']),
            ctrl.Rule(angle['pos big'] & x['left big'], v['very slow']),
            ctrl.Rule(angle['pos big'] & x['left small'], v['slow']),
            ctrl.Rule(angle['pos big'] & x['center'], v['medium']),
            ctrl.Rule(angle['pos big'] & x['right small'], v['fast']),
            ctrl.Rule(angle['pos big'] & x['right big'], v['very fast']),
            ctrl.Rule(angle['pos small'] & x['left big'], v['slow']),
            ctrl.Rule(angle['pos small'] & x['left small'], v['medium']),
            ctrl.Rule(angle['pos small'] & x['center'], v['fast']),
            ctrl.Rule(angle['pos small'] & x['right small'], v['very fast']),
            ctrl.Rule(angle['pos small'] & x['right big'], v['very fast']),
            ctrl.Rule(angle['neutr'] & x['left big'], (v['fast'], w['pos_fast'])),
            ctrl.Rule(angle['neutr'] & x['left small'], (v['very fast'], w['pos_slow'])),
            ctrl.Rule(angle['neutr'] & x['center'], (v['very fast'], w['neutr'])),
            ctrl.Rule(angle['neutr'] & x['right small'], (v['very fast'], w['neg_slow'])),
            ctrl.Rule(angle['neutr'] & x['right big'], (v['fast'], w['neg_fast'])),
            ctrl.Rule(angle['neg small'] & x['left big'], v['very fast']),
            ctrl.Rule(angle['neg small'] & x['left small'], v['very fast']),
            ctrl.Rule(angle['neg small'] & x['center'], v['fast']),
            ctrl.Rule(angle['neg small'] & x['right small'], v['medium']),
            ctrl.Rule(angle['neg small'] & x['right big'], v['slow']),
            ctrl.Rule(angle['neg big'] & x['left big'], v['very fast']),
            ctrl.Rule(angle['neg big'] & x['left small'], v['fast']),
            ctrl.Rule(angle['neg big'] & x['center'], v['medium']),
            ctrl.Rule(angle['neg big'] & x['right small'], v['slow']),
            ctrl.Rule(angle['neg big'] & x['right big'], v['very slow']),
        ]
        sy = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(sy)

    def controller_output(self, angle_val, x_val):
        """
        Function to get the fuzzy controller output given an angle and x position.
        Returns the calculated linear and angular velocities.
        """
        self.controller.input['angle'] = angle_val
        self.controller.input['x'] = x_val
        self.controller.compute()
        return self.controller.output['v'], self.controller.output['w']

    def get_line(self, mask, img):
        """
        Function to detect the line in an image given a mask.
        Returns the initial and final points of the detected line.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_angle = None
        best_p_i = [0, 0]
        best_p_f = [0, 0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.area_min_line:
                cv2.drawContours(img, cnt, -1, (255, 0, 255), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(img, f"({cx}, {cy})", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    x_min = min(cnt[:, 0, 0])
                    x_max = max(cnt[:, 0, 0])
                    if x_min >= (self.w_img / 2 - self.w_t) and x_max <= (self.w_img / 2 + self.w_t):
                        y_min = min(cnt[:, 0, 1])
                        y_max = max(cnt[:, 0, 1])
                        x_i = [v[0, 0] for v in cnt if v[0, 1] == y_min]
                        x_f = [v[0, 0] for v in cnt if v[0, 1] == y_max]
                        x1 = int(np.rint((min(x_i) + max(x_i)) / 2))
                        x2 = int(np.rint((min(x_f) + max(x_f)) / 2))
                        y1 = int(y_min)
                        y2 = int(y_max)
                        p_i = [x1, y1]
                        p_f = [x2, y2]
                        angle = self.get_angle(p_i, p_f)
                       
                        if best_angle is None or abs(angle) < abs(best_angle):
                            best_angle = angle
                            best_p_i = p_i
                            best_p_f = p_f
        return best_p_i, best_p_f

    def get_area(self, contours):
        """
        Function to calculate the total area of the detected contours.
        """
        a = 0
        for cnt in contours:
            a += cv2.contourArea(cnt)
        return a

    def get_angle(self, p_i, p_f):
        """
        Function to calculate the angle between two points.
        """
        y = p_f[0] - p_i[0]
        x = p_f[1] - p_i[1]
        angle = np.arctan2(y, x)
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        return angle

    def camera_callback(self, msg1):
        """
        Camera subscription callback function.
        Converts the camera message to an OpenCV image and processes it.
        """
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg1, 'bgr8')
            self.img = cv2.rotate(self.img, cv2.ROTATE_180)
            self.img1 = self.img.copy()[10:240, 0:320]
            self.img2 = self.img.copy()[160:240, 0:320]
            self.image_received = True
        except Exception as e:
            self.get_logger().error(f'Failed to get image: {e}')
            self.image_received = False
        
    def detected_object_callback(self, msg):
        """
        Detected object subscription callback function.
        Updates the detected objects dictionary with the current time.
        """
        detected_object = msg.data
        self.get_logger().info(f'Detected object: {detected_object}')
        self.detected_objects[detected_object] = time.time()
        if detected_object:
            self.stop_robot = 0
        
        elif detected_object == 'left' or detected_object == 'right':
            self.c2 = .8
        """
        if detected_object == "Green" or self.semaforo == 3:
            self.val_semaforo = 1.0
        elif detected_object == "Yellow" or self.semaforo == 2:
            self.val_semaforo = 0.5
        elif detected_object == "Red" or self.semaforo == 1:
            self.val_semaforo = 0.0
        self.last_signal = detected_object  # Store the last detected signal
        """
        
    def restore_velocity(self):
        """
        Function to restore the robot's original speed.
        """
        self.get_logger().info('Restoring Speed')
        self.robot_vel.linear.x = 0.1
        self.robot_vel.angular.z = 0.0
        self.pub_cmd_vel.publish(self.robot_vel)
        self.velocity_reduced = False
        self.stop_robot = 0
    
    def arrancar(self):
        """
        Function to start the robot.
        """
        self.get_logger().info('Starting')
        self.robot_vel.linear.x = 0.2
        self.robot_vel.angular.z = 0.0
        self.pub_cmd_vel.publish(self.robot_vel)
        self.velocity_reduced = False
        self.stop_robot = 0

    def analyze_signal_area(self, mask):
        """
        Function to analyze the detected signal and determine which side is darker.
        Returns 'left' if the left side is darker, 'right' if the right side is darker.
        """
        left_mask = mask[:, :self.w_t]
        right_mask = mask[:, self.w_t:]
        left_area = self.get_area(cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        right_area = self.get_area(cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
        if left_area > right_area:
            return 'left'
        else:
            return 'right'

    def timer_callback(self):
        """
        Timer callback function. Processes the image, detects lines and objects, and publishes movement commands.
        """
        if self.image_received:
            self.img_mod1 = self.img1.copy()
            self.img_mod2 = self.img2.copy()
            hsv1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2HSV)

            min_black = np.array([0, 0, 0])
            max_black = np.array([255, 85, 85])

            mask_black = cv2.inRange(hsv2, min_black, max_black)

            p_i, p_f = self.get_line(mask_black.copy(), self.img_mod2)

            cv2.line(self.img_mod2, p_i, p_f, (255, 255, 0), 2)
            cv2.circle(self.img_mod2, p_f, 5, (0, 255, 0), -1)
            self.detect_line = (p_i != [0, 0]) and (p_f != [0, 0])
            self.navigation(p_i, p_f)
            self.pub_image1.publish(self.bridge.cv2_to_imgmsg(self.img_mod1, 'bgr8'))
            self.pub_image2.publish(self.bridge.cv2_to_imgmsg(self.img_mod2, 'bgr8'))

            current_time = time.time()
        
            timeout = 1
            timeout3 = 2 # 1 second timeout

            if 'stop' in self.detected_objects:
                last_detected = self.detected_objects['stop']
                if current_time - last_detected > timeout:
                    self.get_logger().info('Stop Signal Detect')
                    self.stop_robot = 1
                    self.last = time.time()

            elif 'works' in self.detected_objects:
                last_detected = self.detected_objects['works']
                if current_time - last_detected > timeout:
                    self.get_logger().info('Work Signal Detect')
                    self.stop_robot = 2
                    self.last = time.time()

            elif 'give_way' in self.detected_objects:
                last_detected = self.detected_objects['give_way']
                if current_time - last_detected > timeout:
                    self.get_logger().info('Giveway Signal Detect')
                    self.stop_robot = 3
                    self.last = time.time()
            
            elif 'straight' in self.detected_objects:
                last_detected = self.detected_objects['straight']
                if current_time - last_detected > timeout:
                    self.get_logger().info('Straight Signal Detect')
                    self.stop_robot = 4
                    self.last = time.time()

            # Perform turn signal analysis
            signal_direction = self.analyze_signal_area(mask_black)

            if 'right' in self.detected_objects:
                last_detected = self.detected_objects['right']
                if current_time - last_detected > timeout3:
                    self.get_logger().info('Right Signal Detect')
                    self.stop_robot = 5
                    self.last = time.time()
                    self.turn_start_time = time.time()  # Save turn start time

            elif 'left' in self.detected_objects:
                last_detected = self.detected_objects['left']
                if current_time - last_detected > timeout3:
                    self.get_logger().info('Left Signal Detect')
                    self.stop_robot = 6
                    self.turn_start_time = time.time()  # Save turn start time

            # Check if traffic light changed from red to green
            if self.previous_val_semaforo == 0.0 and self.val_semaforo == 1.0:
                self.get_logger().info('Traffic light changed from red to green')
                self.stop_robot = self.last_stop_robot  # Restore the last stop state

            self.previous_val_semaforo = self.val_semaforo  # Update previous traffic light value

            if self.semaforo == 0: 
                if self.stop_robot == 1:
                    timeout2 = 10
                    if current_time - self.last < timeout2:   
                        self.get_logger().info('Stopping')

                elif self.stop_robot == 2:
                    timeout2 = 5
                    if current_time - self.last < timeout2:   
                        self.get_logger().info('Slowing Workers')
                    else:
                        self.restore_velocity()    

                elif self.stop_robot == 3:
                    timeout2 = 2
                    if current_time - self.last < timeout2:   
                        self.get_logger().info('Slowing Giveway')
                    else:
                        self.restore_velocity()

                elif self.stop_robot == 4:
                    timeout2 = 2
                    if current_time - self.last < timeout2:   
                        self.get_logger().info('Moving') 
                    else:
                        self.restore_velocity()
                
                elif self.stop_robot == 5:
                    timeout2 = 5
                    if current_time - self.last < timeout2:
                        self.get_logger().info('Turning Right')
 # Turn right
                    else:
                        self.restore_velocity()

                elif self.stop_robot == 6:
                    timeout2 = 5
                    if current_time - self.last < timeout2:  # Increase turn delay
                        self.get_logger().info('Turning Left')
# Turn left
                    else:
                        self.restore_velocity()

                else:
                    self.robot_vel.linear.x = 0.1
                    self.robot_vel.angular.z = 0.0

            objects_to_remove = []
        
            for obj, last_detected in self.detected_objects.items():
                if current_time - last_detected > timeout:
                    self.get_logger().info(f'{obj} has not been detected for {timeout} seconds.')
                    objects_to_remove.append(obj)

            for obj in objects_to_remove:
                del self.detected_objects[obj]

    def navigation(self, p_i, p_f):
        """
        Function for robot navigation based on the detected line.
        """
        if self.detect_line:
            self.move = True
            self.t_detect = time.time() * 1000 // 1
            self.error_line = self.w_img / 2 - p_f[0]
            self.x_line = - self.error_line
            self.angle = self.get_angle(p_i, p_f)
            cv2.putText(self.img_mod2, (str(np.round(self.angle, 4)) + "°"), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            if self.move:
                self.t_now = time.time() * 1000 // 1
                dt = (self.t_now - self.t_detect) / 1000
                self.move = dt <= 1.0

        if self.move:
            self.f_v, self.f_w = self.controller_output(np.rad2deg(self.angle), self.x_line)
            self.robot_vel.linear.x = self.f_v
            self.robot_vel.angular.z = self.f_w

            if self.stop_robot == 1:
                self.get_logger().info('STOP')
                self.robot_vel.linear.x = 0.0
                self.robot_vel.angular.z = 0.0

            elif self.stop_robot == 2: 
                self.get_logger().info('WORK')
                self.robot_vel.linear.x *= 0.4  # Reduce linear speed by half
                self.robot_vel.angular.z *= 0.4  # Reduce angular speed by half
            
            elif self.stop_robot == 3:
                self.get_logger().info('GIVEWAY')
                self.robot_vel.linear.x *= 0.4  # Reduce linear speed by half
                self.robot_vel.angular.z *= 0.4

            elif self.stop_robot == 4:
                self.robot_vel.linear.x *= 1.2  # Increase linear speed by 20%
                self.robot_vel.angular.z *= 1.2 

            elif self.stop_robot == 5:
                self.get_logger().info('TURNING RIGHT')
                self.robot_vel.linear.x = 0.0
                self.robot_vel.angular.z = -2.  # Turn right

            elif self.stop_robot == 6:
                self.get_logger().info('TURNING LEFT')
                self.robot_vel.linear.x = 0.0
                self.robot_vel.angular.z = 2.  # Turn left

            elif self.stop_robot == 9:
                self.get_logger().info('Bajando velocidad')
                self.robot_vel.linear.x *= 0.6
                self.robot_vel.angular.z = 0.0  #

        else:
            self.robot_vel.linear.x = 0.1
            self.robot_vel.angular.z = 0.0

        # Adjust speed based on traffic light state
        self.robot_vel.linear.x *= self.val_semaforo
        self.robot_vel.angular.z *= self.val_semaforo
        self.pub_cmd_vel.publish(self.robot_vel)

def main(args=None):
    """
    Main function that initializes the ROS 2 node and runs it.
    """
    rclpy.init(args=args)
    n_v = VisionClass()
    rclpy.spin(n_v)
    n_v.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

