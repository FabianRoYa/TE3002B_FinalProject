#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from yolo_msg.msg import InferenceResult, Yolov8Inference
from std_msgs.msg import String

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.model = YOLO('red.pt')
        self.yolov8_inference = Yolov8Inference()
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)   ####Run in puzzlebot
        #self.subscription = self.create_subscription(Image, 'image_raw', self.camera_callback, 10)          
        self.image_width = 640
        self.image_height = 480
        self.img = np.ndarray((self.image_height, self.image_width, 3))
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.detected_object_pub = self.create_publisher(String, "/detected_object", 1)  # Publisher for detected object name
        timer_period = 0.2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        results = self.model(self.img)
        detections_exist = False  # Flag to check if there are any detections
        
        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                detections_exist = True  # Set the flag if there are detections
            for box in boxes:
                self.inference_result = InferenceResult()
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                class_name = self.model.names[int(c)]
                self.inference_result.class_name = class_name
                self.inference_result.top = int(b[1])
                self.inference_result.left = int(b[0])
                self.inference_result.bottom = int(b[3])
                self.inference_result.right = int(b[2])
                self.yolov8_inference.yolov8_inference.append(self.inference_result)
                
                # Publish the detected object name
                detected_object_msg = String()
                detected_object_msg.data = class_name
                self.detected_object_pub.publish(detected_object_msg)

        annotated_frame = results[0].plot() if detections_exist else self.img

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.img_pub.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")

        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

    def camera_callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Optional: Rotate image if necessary
        img = cv2.rotate(img, cv2.ROTATE_180)

        scale = 4
        w = int(self.image_width / scale)
        h = int(self.image_height / scale)
        small = (w, h)
        self.img = cv2.resize(img, small)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
