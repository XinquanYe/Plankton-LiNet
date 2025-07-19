import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/Plankton-LiNet.yaml')
    model.train(data='datasets/data.yaml',
                imgsz=640,
                epochs=200,
                )