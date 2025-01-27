#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from ultralytics import YOLO
import os

class MyNode(Node):

    def __init__(self):
        super().__init__("img_sub")
        self.cv_image = None
        self.bridge = CvBridge()
        self.carpeta = 'results'
        self.frame_num = 0
        self.msg = True
        os.makedirs(self.carpeta, exist_ok=True)
        self.model = YOLO('xlarge1280.pt')   #xlarge tarda entre 9 y 18 segundos por imagen, nano entre 0.8 i 1.5
        self.img_sub_ = self.create_subscription(
            Image, "/shot_pub", self.img_callback, 10)
        self.palette = [
            (255, 0, 0),    # Rojo
            (0, 255, 0),    # Verde
            (0, 255, 255),  # Amarillo
            (0, 0, 255),    # Azul
        ]
        self.labels = ["Bottom", "Top", "Mid-high", "Mid-low"]

    # Función para dibujar keypoints con colores y etiquetas en una imagen
    def draw_colored_keypoints(self, image, keypoints):
        for i, keypoint in enumerate(keypoints):
            try:
                x, y = keypoint[:2]
                color = self.palette[i % len(self.palette)]
                label = self.labels[i % len(self.labels)]
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                cv2.putText(image, label, (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Error al dibujar keypoint {i}: {e}")

    
    def img_callback(self, msg: Image):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.get_logger().info("Se ha recibido algo") 
        results = self.model(self.cv_image, conf = 0.3)
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None and keypoints.xy is not None:
                for keypoints_set in keypoints.xy:  # Iterar sobre cada conjunto de keypoints
                    if keypoints_set.shape[0] > 0:  # Comprobar que hay keypoints
                        self.draw_colored_keypoints(self.cv_image, keypoints_set)
                        file_name = f'result_{self.frame_num:04d}.jpg'
                        file_name = os.path.join(self.carpeta, file_name)
                        cv2.imwrite(file_name, self.cv_image)

        self.frame_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
