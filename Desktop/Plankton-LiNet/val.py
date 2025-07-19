import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('test/best.pt')
    metrics=model.val(data='datasets/data.yaml',
              split='val',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True,
              save_json=False,  
              save_hybrid=False,  
              conf=0.55, 
              iou=0.6,  
              project='runs/val', 
              name='exp', 
              max_det=300,  
              half=False,  
              dnn=False, 
              plots=True,  
    )