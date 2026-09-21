import os
from ultralytics import YOLO

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    model = YOLO("yolo11s.pt")
    model.train(
        data="../Datasets/combined.yaml",
        epochs=300,
        patience=50,
        imgsz=640,
        batch=48,
        name="v2",
        project=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "train"),
    )


if __name__ == "__main__":
    main()
