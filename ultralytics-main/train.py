from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    #
    # # Use the model
    #
    # model.train(data="mycoco128.yaml", epochs=100)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml").train(**{'cfg': 'ultralytics/cfg/default.yaml'})
