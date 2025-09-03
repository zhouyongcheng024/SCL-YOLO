import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/Add/yolov11-SCL.yaml')
    model.train(
        data="./datasets/images",
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=64,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )

