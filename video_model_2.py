#!/usr/bin/env python3

import rclpy
import cv2
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from threading import Thread, Event
from queue import Queue
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
from rich import print as print
import time
import json


class MyNode(Node):

    def __init__(self):
        super().__init__("mod_pub")
        self.bridge = CvBridge()
        
        self.img_pub_ = self.create_publisher(Image, "/image_raw", 10)
        self.box_pub_ = self.create_publisher(String, '/topic', 10)

        FRAME_PASS = 10
        self.debug = True
        print('run')
        self.frame_to_ml = Queue()
        self.ros_queue = Queue()
        # show_queue = Queue()
        # Load a model
        model = YOLO('best_nano.pt')  # pretrained YOLOv8n model
        print('model load')
        source = '/home/jb_07/tfg_jason/tfg_def/SARTI_seguimiento_posidonia_20240509.mp4'  #video plantació
        #source = '/home/jb_07/tfg_jason/tfg_def/OBSEA_posidonia.mp4' #video primer
        # source = 'rtsp://admin:seaslag2024@192.168.102.64:554'  #camara despatx dan
        #source = 'rtsp://admin:sartisarti.@192.168.3.110:554'  #camara crawler
        self.cap = cv2.VideoCapture(source)
        p1 = Thread(target=self.procesar_frame, args=(model, self.frame_to_ml, self.ros_queue, FRAME_PASS, self.debug))
        p1.start()

        p2 = Thread(target=self.bucle_model)
        p2.start()


    def print_debug(self, msg: str, color: str, debug_bool):
        if debug_bool:
            print(f'[{color}]{datetime.now().strftime("%H:%M:%S.%f")} - {msg}')

        

    def bucle_model(self):
        while True:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                # cv2.imshow('frame', frame)
                if ret:
                    self.frame_to_ml.put(frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.cap.release()
            else:
                self.cap.release()
                    

    def send_data(self):
        # folder = '/home/albert/srv/ultralytics/results'
        # msg = cv2.imread(os.path.join(folder, 'result.jpg'))
        if not self.ros_queue.empty():
            (ann_img, dict_res) = self.ros_queue.get()
            self.img_pub_.publish(self.bridge.cv2_to_imgmsg(ann_img, "bgr8"))
            msg = String()
            msg.data = dict_res
            self.box_pub_.publish(msg)
            print("Se ha publicado algo")

    
    def procesar_frame(self, model, in_queue: Queue, out_queue: Queue, frames_pass_inf: int, debug: bool = False):

        count = 0
        result = None
        frame = None
        t = None
        out_list = []
        while True:
            while in_queue.qsize() > 0:
                self.print_debug(msg=f'(INF)Got {in_queue.qsize()} frames in queue', color='cyan', debug_bool=self.debug)
                if in_queue.qsize() > 5:
                    self.print_debug(msg='(INF)EMPTY BUFFER!', color='yellow', debug_bool=self.debug)
                    while in_queue.qsize() > 1:
                        frame = in_queue.get()
                        in_queue.task_done()

                print(count)
                frame = in_queue.get()

                if count == 0:
                    t = time.time()

                    results = model(frame, imgsz=384, verbose=False)
                    result = results[0]
                    annotated_frame = result.plot(line_width=2)
                    self.print_debug(msg=f'(INF) inference time {1000*(time.time() - t):.4f} msecs', color='yellow', debug_bool=self.debug)

                    # out_list = []
                    # for box in result.boxes:
                    #     print_debug(msg=f'(INF) clase detecció: {box.cls}, box confidence: {box.conf}, '
                    #                     f'position xywhn {box.xywhn}', color='green', debug_bool=debug)
                    #     dict_box = {'class': box.cls, 'conf': box.conf, 'xywhn': box.xywhn}
                    #     out_list.append(dict_box)
                    out_list = [{"cantidad": len(result.boxes), "box_data": {"class": float(box.cls), "conf": float(box.conf), "xywhn": box.xywhn.tolist()}} for box in result.boxes]
                    out_list = json.dumps(out_list)
                    self.print_debug(msg=out_list, color='red', debug_bool=self.debug)

                    out_queue.put((annotated_frame, out_list))

                else:
                    self.print_debug(msg=f'Count =! {count}', color='red', debug_bool=self.debug)
                    t = time.time()

                    result.orig_img = frame
                    self.print_debug(msg=f'(INF) overwrite time {1000*(time.time() - t):.4f} msecs', color='cyan', debug_bool=self.debug)

                count += 1

                if count == frames_pass_inf:
                    count = 0

                in_queue.task_done()
                self.send_data()

            time.sleep(0.0001)


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
