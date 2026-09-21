# Phantom braking: teaching a pedestrian detector what isn't a pedestrian

Code for my bachelor's thesis at Halmstad University (2026), *A Conceptual Framework for Enhancing Pedestrian Detection and Reducing Phantom Braking through Robust Object Classification*.

Emergency braking systems that rely on a camera can mistake statues, mannequins, and other human-shaped objects for pedestrians and brake for no reason. The idea is simple: instead of training the detector only on people, add a second class, `person-like`, and let a small rule-based monitor decide when a detection is actually worth braking for.

On the validation data, the confuser-trained YOLO11s cut the phantom brake rate from 47.92% to 1.04% compared to stock YOLO11s, while the real-pedestrian brake rate went from 70.30% to 70.15%. The validation images come from the same sources as the training images, so treat those numbers as an upper bound, not as road performance.

## What's in here

| Folder | What it is |
| --- | --- |
| `Monitor/` | The runtime monitor. Runs the detector, tracks objects across frames, and outputs a decision per object. Has a Tkinter GUI and a headless mode. |
| `YOLO11/` | Training script for YOLO11s and the final run results (`runs/train/v2`). |
| `YOLOv5/` | A copy of [ultralytics/yolov5](https://github.com/ultralytics/yolov5) used for the first experiments, plus two training runs (`v1` without Cityscapes, `v2` with it). The only upstream change is that autocast uses `"cuda"` instead of the deprecated call. |
| `Labellmg/` | The class list used when hand-labeling images in LabelImg. |

The two classes are:

```
0  person
1  person-like   (statues, sculptures, mannequins, costumed performers)
```

## Setup

Python 3.10 or newer. I used Python 3.13, PyTorch 2.6 with CUDA 12.4 and Ultralytics 8.4 on an RTX 4070 Ti SUPER, but any recent versions should work.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python pillow numpy
```

If you only want to run the monitor on the CPU, a plain `pip install torch` is enough.

YOLOv5 has its own dependencies if you want to train with it:

```bash
pip install -r YOLOv5/requirements.txt
```

## Running the monitor

The trained YOLO11 weights are in `YOLO11/runs/train/v2/weights/best.pt`. This is the model the thesis results are based on.

GUI:

```bash
python Monitor/app.py
```

Pick the weights file, then choose a video, a webcam index, or a batch of images. Images are played back one every 1.5 seconds so you have time to read each decision. You can also save the annotated output to an mp4.

Headless:

```bash
python Monitor/app.py --cli path/to/video.mp4 --weights YOLO11/runs/train/v2/weights/best.pt --save-video out.mp4
```

`--cli` also takes a webcam index, e.g., `--cli 0`. Without `--weights`, it falls back to the YOLOv5 v2 weights set in `Monitor/config.py`.

Ultralytics models are loaded through the `ultralytics` package. If that fails, the monitor tries to load the weights as a YOLOv5 model through the local `YOLOv5/` folder, so both sets of weights work.

### How the monitor decides

Each detection above 0.5 confidence is matched to a track by IoU (threshold 0.4). A track keeps the last 10 frames of class, confidence and box position, and is dropped after 5 frames without a match. From that history each track gets one of four labels:

- **BRAKE**: consistently a `person`, close to the car and in front of it.
- **SUPPRESS**: consistently `person-like`, and either stationary or detected with high confidence (0.786 or above). No braking for this object.
- **WARN**: the class keeps flipping between `person` and `person-like`, or it's a pedestrian that is too far away or off to the side to brake for.
- **NORMAL**: the track is too unstable to trust, so the monitor stays out of it and the normal collision avoidance handles it.

A change to BRAKE happens immediately. Every other change has to hold for 5 frames before it's applied, so a single bad frame doesn't cause a flicker. All the thresholds live in `Monitor/config.py`.

## Datasets

This repo doesn't include the datasets. Most of their licenses don't allow redistribution, and together they're around 20 000 images. You'll have to download them yourself:

| Dataset | Used as | Where to get it | Licence |
| --- | --- | --- | --- |
| Person and person-like objects (PnPLO), by Karthika95 | `person` and `person-like` | [Kaggle](https://www.kaggle.com/datasets/karthika95/pedestrian-detection) | See Kaggle page |
| Cityscapes | `person` (road scenes) | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/), needs a free account. You want `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`. | Non-commercial research only |
| Sculptures 6k | `person-like` | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/sculptures6k/) (the original download links are down) or the [Kaggle mirror](https://www.kaggle.com/datasets/hunter0007/6k-sculptures-dataset) | See source |
| Original Statues v2 ("Originals augmented") | `person-like` | [Roboflow Universe](https://universe.roboflow.com/statue-artworks/original-statues), export as YOLOv5 PyTorch | CC BY 4.0 |
| Human-Art (sculpture, cosplay, dance, drama scenes) | `person` and `person-like` | [IDEA-Research/HumanArt](https://github.com/IDEA-Research/HumanArt). Fill in the request form, and the download link will be emailed to you. | Non-commercial research only |

A few notes on preparing them:

- **PnPLO** ships as Pascal VOC XML. Its class names are already `person` and `person-like`, so converting to YOLO format is a straight box conversion (`x_center y_center width height`, normalized).
- **Sculptures 6k** has no bounding boxes for this task. I labeled the images myself in [LabelImg](https://github.com/HumanSignal/labelImg) with the classes in `Labellmg/classes.txt`.
- **Original Statues** has its own class list. Remap every statue class to `1` (`person-like`).
- **Cityscapes** provides polygons, not boxes. Take the `person` instances from the `gtFine` polygon files and turn each one into a bounding box with class `0`.
- **Human-Art** annotations are COCO-style JSON. Pick the scenes you need and write one YOLO label file per image.

Every dataset should end up in the usual YOLO layout:

```
Datasets/
  <dataset>/
    train/images/   train/labels/
    val/images/     val/labels/
```

Then create `Datasets/combined.yaml` pointing to all of them. Paths are relative to the YAML file's location or absolute:

```yaml
train:
  - archive/train/images
  - cityscapes/train/images
  - sculptures6k/train/images
  - Original Statues.v2-originals-augmented.yolov5pytorch/train/images
  - HumanArtv1/sculptures/train/images
  - HumanArtv1/cosplay/train/images
  - HumanArtv1/dance/train/images
  - HumanArtv1/drama/train/images

val:
  - archive/val/images
  - cityscapes/val/images
  - sculptures6k/val/images
  - Original Statues.v2-originals-augmented.yolov5pytorch/valid/images

nc: 2
names: ['person', 'person-like']
```

Human-Art is intentionally left out of validation. It mixes real people in costumes with actual sculptures, which makes the per-class numbers hard to read.

## Training

YOLO11 (the final model):

```bash
cd YOLO11
python train.py
```

That trains `yolo11s.pt` for up to 300 epochs, with early stopping after 50 epochs without improvement, image size 640, and batch size 48. Output goes to `YOLO11/runs/train/<name>`. Lower the batch size if you run out of VRAM. The pretrained `yolo11s.pt` downloads automatically the first time.

YOLOv5:

```bash
cd YOLOv5
python train.py --img 640 --batch 16 --epochs 300 --data ../Datasets/combined.yaml --weights yolov5s.pt --name v2
```

## Results

YOLOv5s vs YOLO11s, same data and settings:

| Metric | YOLOv5s | YOLO11s |
| --- | --- | --- |
| mAP@0.5 | 0.7315 | 0.7394 |
| mAP@0.5:0.95 | 0.4830 | 0.5188 |
| Precision | 0.7883 | 0.8061 |
| Recall | 0.6391 | 0.6516 |
| Epochs run | 300 | 129 (best at 76) |

Brake decisions from the monitor, stock YOLO11s against the confuser-trained model:

| | Stock YOLO11s | Confuser-trained YOLO11s |
| --- | --- | --- |
| Phantom brake rate (images with no real pedestrian) | 47.92% | 1.04% |
| Real-pedestrian brake rate | 70.30% | 70.15% |

Full training curves and confusion matrices are in each `runs/train/*` folder.

## Limitations

- Nothing here has been tested in a car or a driving simulator. We evaluated it by running validation images through the GUI.
- Validation and training data come from the same sources. Expect worse numbers on anything new.
- Cityscapes is all European cities, and the statue data doesn't cover much variety. Confusers that look nothing like the training data still get through.
- Some Human-Art cosplay images show real people dressed as statues and are labeled `person-like`. That's likely why real-pedestrian braking drops slightly.

## Acknowledgements

I sincerely thank my supervisor, Alexandre dos Santos Roque, for guidance, technical insight, and continuous support throughout this project.

## Licence

The `YOLOv5/` folder is Ultralytics code under AGPL-3.0 (see `YOLOv5/LICENSE`), and YOLO11 is used through the `ultralytics` package, which is also AGPL-3.0. Each dataset has its own terms, listed above.
