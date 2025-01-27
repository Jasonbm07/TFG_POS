#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rich import print as print
import json
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time

class SILVERMovement(Node):

    def __init__(self):
        super().__init__("movement_node")
        self.cv_image = None
        self.bridge = CvBridge()
        self.is_moving = False
        self.previous_action = ''
        self.target_shot = None

        # Variables específicas para SILVER 2
        self.hopping_forward_distance = 1.0  # Distancia en metros para "hopping forward"
        self.walking_speed = 0.5  # Velocidad para "walking locomotion" (m/s)

        # Suscriptores y publicadores
        self.img_pub_ = self.create_publisher(Image, "/shot_pub", 10)
        self.img_sub_ = self.create_subscription(
            Image, "/image_raw", self.img_callback, 10)
        self.box_sub_ = self.create_subscription(
            String, "/topic", self.box_callback, 10)

    def img_callback(self, msg: Image):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def box_callback(self, msg: String):
        n_msg = json.loads(msg.data)
        print(n_msg)

        if len(n_msg) != 0:
            max_area = 0
            target_index = None

            for i, box in enumerate(n_msg):
                if box['box_data']['class'] == 2.0:  # Clase que identifica "shot" de Posidonia
                    area = box['box_data']['xywhn'][0][2] * box['box_data']['xywhn'][0][3]
                    if area > max_area:
                        max_area = area
                        target_index = i

            if target_index is not None:
                self.target_shot = n_msg[target_index]
                print("Shot identificado:", self.target_shot)
                self.decide_movement()
        else:
            self.perform_hopping_forward()

    def perform_hopping_forward(self):
        print("No se detectaron shots. Realizando movimiento 'hopping forward'.")
        self.is_moving = True
        self.execute_hopping_forward(self.hopping_forward_distance)
        self.is_moving = False

    def execute_hopping_forward(self, distance):
        print(f"Moviendo hacia adelante mediante 'hopping forward' por {distance} metros.")
        # Aquí se implementaría el código para controlar el movimiento de hopping forward del robot.
        time.sleep(distance / self.walking_speed)  # Simula tiempo del movimiento

    def decide_movement(self):
        if self.target_shot['box_data']['xywhn'][0][1] >= 0.9:
            self.is_moving = True
            print("Acción: NO_PISAR")
            self.previous_action = "NO_PISAR"
            self.is_moving = False

        elif self.target_shot['box_data']['xywhn'][0][0] <= 0.25 and not self.is_moving:
            self.is_moving = True
            print("Acción: girar_derecha")
            self.perform_turn("derecha")
            self.previous_action = "girar_derecha"
            self.is_moving = False

        elif self.target_shot['box_data']['xywhn'][0][0] >= 0.75 and not self.is_moving:
            self.is_moving = True
            print("Acción: girar_izquierda")
            self.perform_turn("izquierda")
            self.previous_action = "girar_izquierda"
            self.is_moving = False

        elif self.target_shot['box_data']['xywhn'][0][1] <= 0.25 and not self.is_moving:
            self.is_moving = True
            print("Acción: acercarse al shot usando 'walking locomotion'")
            self.perform_walking_loc()  # Método para caminar hacia el shot
            self.previous_action = "walking_loc"
            self.is_moving = False

        else:
            self.publish_shot_image()

    def perform_turn(self, direction):
        print(f"Realizando giro hacia la {direction}.")
        # Implementar lógica de control para giros del robot.
        time.sleep(1)  # Simula tiempo del giro

    def perform_walking_loc(self):
        print("Acercándose al shot usando 'walking locomotion'.")
        distance = (1 - self.target_shot['box_data']['xywhn'][0][1]) * 2.0  # Distancia aproximada al shot
        print(f"Caminando {distance} metros hacia el shot.")
        time.sleep(distance / self.walking_speed)  # Simula tiempo del movimiento

    def publish_shot_image(self):
        if self.cv_image is not None:
            h, w, c = self.cv_image.shape
            ROI = self.cv_image[
                int((self.target_shot['box_data']['xywhn'][0][1] - self.target_shot['box_data']['xywhn'][0][3] / 2) * h):
                int((self.target_shot['box_data']['xywhn'][0][1] + self.target_shot['box_data']['xywhn'][0][3] / 2) * h),
                int((self.target_shot['box_data']['xywhn'][0][0] - self.target_shot['box_data']['xywhn'][0][2] / 2) * w):
                int((self.target_shot['box_data']['xywhn'][0][0] + self.target_shot['box_data']['xywhn'][0][2] / 2) * w)
            ]
            cv2.imwrite('ROI.jpg', ROI)
            self.img_pub_.publish(self.bridge.cv2_to_imgmsg(ROI, 'bgr8'))
            print("Imagen del shot publicada.")


def main(args=None):
    rclpy.init(args=args)
    node = SILVERMovement()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
