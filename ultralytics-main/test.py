from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")
model = YOLO('runs/detect/train4/weights/best.pt')  # pretrained YOLOv8n model
# results = model("dataset/data/images/test")
# 模型预测，save=True 的时候表示直接保存yolov8的预测结果
# metrics = model.predict(source='dataset/data/a', save=True, hide_labels=True)
metrics = model.predict(source='dataset/data/a')
print(metrics)

# for m in metrics:
#     # 获取每个boxes的结果
#     box = m.boxes
#     # 获取box的位置，
#     xywh = box.xywh
#     # 获取预测的类别
#     cls = box.cls
#
#     print(box, xywh, cls)
# # Run batched inference on a list of images
# results = model(['dataset/data/images/test/IMG_0040.jpg'])  # return a list of Results objects
#
# # Process results list
# for result in metrics:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs



