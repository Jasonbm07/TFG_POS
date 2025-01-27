from PIL import Image
from ultralytics import YOLO
import os
from rich import print as print
import cv2

#model = YOLO('best.pt')
model = YOLO('xlarge1280.pt')
results = model('poseidon.jpg', conf=0.1)  # results list
for r in results:

    im = r.plot(line_width=2)

    cv2.imwrite('result3.jpg', im)



    print(r)
    print(len(r.boxes))
    # for box in r.boxes:
    #     if float(box.cls) == 2.0:
    #         i += 1
    #         print(box.xywhn)
    # print(i)
#"box_data": {"class": float(box.cls), "conf": float(box.conf), "xywhn": box.xywhn.tolist()}} for box in result.boxes]