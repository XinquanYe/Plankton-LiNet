import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('val/best.pt')
    model.export(format='onnx', simplify=True, opset=13)