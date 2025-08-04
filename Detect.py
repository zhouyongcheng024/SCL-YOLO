import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/train/exp23/weights/best.pt') # select your model.pt path
    model.predict(source='./06611.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )
# -53-_jpg.rf.7e12183b7fe6474ec7fc887b16f2e036
# -56-_jpg.rf.0bcbe4977c27f10c62366f643d1d5ee3
# -51-_jpg.rf.6615d7f1f6cf90d25968b0318bf4dd22