#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from rich import print as print
import json
import time

class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')
        self.get_logger().info("Nodo de manipulación iniciado.")

        # Subscriber para recibir datos de keypoints sobre la posición y orientación del shot
        self.keypoints_sub = self.create_subscription(
            String, "/keypoints_data", self.keypoints_callback, 10)

        # Subscriber para recibir imágenes (opcional, si se necesitan para validación visual)
        self.image_sub = self.create_subscription(
            Image, "/shot_pub", self.image_callback, 10)

        # Publisher para enviar comandos al brazo manipulador
        self.manipulator_pub = self.create_publisher(
            Pose, "/manipulator_command", 10)

        # Variables internas
        self.keypoint_data = None
        self.ready_to_manipulate = False

    def keypoints_callback(self, msg):
        """
        Procesa los datos recibidos desde el nodo de keypoints.
        """
        try:
            self.keypoint_data = json.loads(msg.data)  # Parsear la información de keypoints
            self.get_logger().info(f"Datos de keypoints recibidos: {self.keypoint_data}")

            # Comprobar si el shot está mal posicionado
            if self.is_shot_misaligned(self.keypoint_data):
                self.get_logger().info("Shot mal posicionado, preparando para manipulación...")
                self.ready_to_manipulate = True
                self.execute_manipulation(self.keypoint_data)
            else:
                self.get_logger().info("Shot bien posicionado. No se requiere manipulación.")

        except Exception as e:
            self.get_logger().error(f"Error procesando datos de keypoints: {e}")

    def is_shot_misaligned(self, keypoint_data):
        """
        Comprueba si el shot está mal posicionado basándose en los datos de keypoints.
        """
        # Idea: Evaluar si los keypoints están fuera de un rango aceptable
        orientation_threshold = 0.1  # Umbral de orientación aceptable
        if abs(keypoint_data["orientation"]) > orientation_threshold:
            return True
        return False

    def execute_manipulation(self, keypoint_data):
        """
        Envía comandos al brazo manipulador para corregir la posición del shot.
        """
        # Generar la posición de destino basada en los datos de keypoints
        target_pose = Pose()
        target_pose.position.x = keypoint_data["x"]  # Coordenadas del shot
        target_pose.position.y = keypoint_data["y"]
        target_pose.position.z = keypoint_data["z"] + 0.1  # Ajuste en altura para posicionar

        target_pose.orientation.x = 0.0  # Orientación deseada
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 1.0

        # Publicar comando al brazo manipulador
        self.manipulator_pub.publish(target_pose)
        self.get_logger().info(f"Comando enviado al manipulador: {target_pose}")

        # Esperar un tiempo para permitir la manipulación
        time.sleep(2)

        # Comprobación post-manipulación (opcional)
        self.get_logger().info("Manipulación completada. Verificando resultados...")
        self.verify_manipulation()

    def verify_manipulation(self):
        """
        Verifica si la manipulación fue exitosa.
        """
        # Lógica para verificar si el shot ha sido reposicionado correctamente
        if self.keypoint_data and not self.is_shot_misaligned(self.keypoint_data):
            self.get_logger().info("Reposición exitosa.")
        else:
            self.get_logger().warning("Reposición fallida. Reintentando...")
            self.execute_manipulation(self.keypoint_data)

    def image_callback(self, msg):
        """
        Procesa imágenes recibidas (si es necesario para validación).
        """
        self.get_logger().info("Imagen recibida para validación visual.")

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
