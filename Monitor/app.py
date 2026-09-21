import argparse
import ctypes
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
from PIL import Image, ImageTk

from config import CONFIDENCE_THRESH, DEFAULT_WEIGHTS, SETTINGS_FILE, YOLOV5_ROOT
from core import Tracker, decide, draw

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
SECONDS_PER_IMAGE = 1.5
UI_REFRESH_MS = 16
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1


def load_detector(weights_path):
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))

        def predict(frame):
            result = model.predict(frame, conf=CONFIDENCE_THRESH, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            return [
                {
                    "bbox": boxes[i].tolist(),
                    "class_id": int(classes[i]),
                    "confidence": float(confidences[i]),
                }
                for i in range(len(classes))
            ]

        return predict
    except Exception:
        pass

    model = torch.hub.load(
        str(YOLOV5_ROOT), "custom", path=str(weights_path), source="local"
    )
    model.conf = CONFIDENCE_THRESH

    def predict(frame):
        detections = []
        for *bbox, confidence, class_id in model(frame).xyxy[0].cpu().numpy():
            detections.append(
                {
                    "bbox": [float(value) for value in bbox],
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                }
            )
        return detections

    return predict


def _frames_from_image_list(image_paths):
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read '{image_paths[0]}'")
    height, width = first.shape[:2]
    output_fps = 1.0 / SECONDS_PER_IMAGE
    yield first, output_fps, (width, height), True
    for path in image_paths[1:]:
        image = cv2.imread(str(path))
        if image is not None:
            yield image, output_fps, (width, height), True


def _frames_from_capture(source):
    capture = cv2.VideoCapture(
        int(source) if str(source).isdigit() else str(source)
    )
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open '{source}'")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame, fps, (width, height), False
    finally:
        capture.release()


def _frames_from(source):
    if isinstance(source, (list, tuple)):
        yield from _frames_from_image_list(source)
    else:
        yield from _frames_from_capture(source)


def _hold_for_next_image(stop_event):
    deadline = time.perf_counter() + SECONDS_PER_IMAGE
    while time.perf_counter() < deadline:
        if stop_event and stop_event.is_set():
            return
        time.sleep(0.05)


def run(source, weights_path, save_video=None, stop_event=None, frame_callback=None):
    predict = load_detector(weights_path)
    tracker = Tracker()
    writer = None
    frame_index = 0
    last_time = time.perf_counter()

    for frame, output_fps, size, is_still_image in _frames_from(source):
        if stop_event and stop_event.is_set():
            break

        detections = predict(frame)
        tracks = tracker.update(detections)
        frame_height, frame_width = frame.shape[:2]
        for track in tracks:
            decide(track, frame_height, frame_width)

        now = time.perf_counter()
        live_fps = 1.0 / max(now - last_time, 1e-9)
        last_time = now

        annotated = draw(frame, tracks, frame_index, live_fps)

        if writer is None and save_video:
            writer = cv2.VideoWriter(
                save_video, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size
            )
        if writer:
            writer.write(annotated)

        if frame_callback:
            frame_callback(annotated)
        else:
            cv2.imshow("Semantic Safety Monitor", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

        if is_still_image:
            _hold_for_next_image(stop_event)

    if writer:
        writer.release()
    if not frame_callback:
        cv2.destroyAllWindows()


def list_available_cameras(max_index=8):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    cameras = []
    for index in range(max_index):
        capture = (
            cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        )
        if capture.isOpened():
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((index, width, height))
        capture.release()
    return cameras


def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_settings(data):
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


class MonitorGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Semantic Safety Monitor")
        self.root.resizable(False, False)
        self.root.after(0, self._remove_window_icon)

        self.settings = load_settings()
        self.selected_images = []
        self.worker_thread = None
        self.stop_event = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.current_photo = None

        self._build_layout()

    def _remove_window_icon(self):
        window_handle = int(self.root.wm_frame(), 16)
        for icon_type in (ICON_SMALL, ICON_BIG):
            ctypes.windll.user32.SendMessageW(
                window_handle, WM_SETICON, icon_type, 0
            )

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.grid(sticky="nsew")

        self._build_preview_canvas(container, row=0)
        self._build_source_row(container, row=1)
        self._build_weights_row(container, row=2)
        self._build_save_row(container, row=3)
        ttk.Separator(container, orient="horizontal").grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=6
        )
        self._build_control_buttons(container, row=5)
        self._build_status_label(container, row=6)

    def _build_preview_canvas(self, parent, row):
        self.preview = tk.Canvas(
            parent,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="black",
            highlightthickness=0,
        )
        self.preview.grid(row=row, column=0, columnspan=4, pady=(0, 8))
        self.preview_image_id = self.preview.create_image(0, 0, anchor="nw")

    def _build_source_row(self, parent, row):
        ttk.Label(parent, text="Source:").grid(row=row, column=0, sticky="w", padx=8, pady=3)
        self.source = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.source, width=42).grid(
            row=row, column=1, padx=8, pady=3
        )

        button_bar = ttk.Frame(parent)
        button_bar.grid(row=row, column=2, columnspan=2, sticky="w")
        ttk.Button(
            button_bar,
            text="Browse",
            width=8,
            command=lambda: self._pick_file(self.source, "open"),
        ).pack(side="left", padx=2)
        ttk.Button(
            button_bar, text="Images", width=8, command=self._choose_images
        ).pack(side="left", padx=2)
        ttk.Button(
            button_bar, text="Cam", width=6, command=self._choose_camera
        ).pack(side="left", padx=2)

    def _build_weights_row(self, parent, row):
        ttk.Label(parent, text="Weights:").grid(row=row, column=0, sticky="w", padx=8, pady=3)
        self.weights = tk.StringVar(
            value=self.settings.get("weights", str(DEFAULT_WEIGHTS))
        )
        ttk.Entry(parent, textvariable=self.weights, width=42).grid(
            row=row, column=1, padx=8, pady=3
        )
        ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._pick_file(self.weights, "open"),
        ).grid(row=row, column=2, padx=8, pady=3)

    def _build_save_row(self, parent, row):
        ttk.Label(parent, text="Save to:").grid(row=row, column=0, sticky="w", padx=8, pady=3)
        self.save_path = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.save_path, width=42).grid(
            row=row, column=1, padx=8, pady=3
        )
        ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._pick_file(self.save_path, "save"),
        ).grid(row=row, column=2, padx=8, pady=3)

    def _build_control_buttons(self, parent, row):
        button_bar = ttk.Frame(parent)
        button_bar.grid(row=row, column=0, columnspan=4, pady=2)
        self.start_button = ttk.Button(
            button_bar, text="Start", width=12, command=self._on_start_pressed
        )
        self.start_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(
            button_bar,
            text="Stop",
            width=12,
            state="disabled",
            command=self._on_stop_pressed,
        )
        self.stop_button.pack(side="left", padx=6)

    def _build_status_label(self, parent, row):
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_text, foreground="gray").grid(
            row=row, column=0, columnspan=4, pady=(4, 0)
        )

    def _pick_file(self, target_var, mode):
        if mode == "save":
            chosen = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
            )
        else:
            chosen = filedialog.askopenfilename()
        if not chosen:
            return
        target_var.set(chosen)
        if target_var is self.source:
            self.selected_images = []

    def _choose_camera(self):
        self.selected_images = []
        cameras = list_available_cameras()
        if not cameras:
            messagebox.showwarning("No cameras", "No webcams detected.")
            return
        if len(cameras) == 1:
            self.source.set(str(cameras[0][0]))
            return
        self._open_camera_dialog(cameras)

    def _open_camera_dialog(self, cameras):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select camera")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        ttk.Label(dialog, text="Available cameras:", padding=8).pack(anchor="w")

        selected_index = tk.IntVar(value=cameras[0][0])
        for index, width, height in cameras:
            ttk.Radiobutton(
                dialog,
                text=f"Camera {index}  ({width}x{height})",
                variable=selected_index,
                value=index,
            ).pack(anchor="w", padx=16)

        def confirm():
            self.source.set(str(selected_index.get()))
            dialog.destroy()

        button_bar = ttk.Frame(dialog, padding=8)
        button_bar.pack(fill="x")
        ttk.Button(button_bar, text="OK", command=confirm, width=10).pack(
            side="right", padx=4
        )
        ttk.Button(
            button_bar, text="Cancel", command=dialog.destroy, width=10
        ).pack(side="right")

    def _choose_images(self):
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All", "*.*"),
            ]
        )
        if not paths:
            return
        self.selected_images = list(paths)
        self.source.set(f"{len(paths)} image(s) selected")

    def _resolve_source(self):
        if self.selected_images:
            return list(self.selected_images)
        raw = self.source.get().strip()
        return int(raw) if raw.isdigit() else raw

    def _on_start_pressed(self):
        weights_path = Path(self.weights.get().strip())
        if not weights_path.exists():
            messagebox.showerror("Error", f"Weights not found:\n{weights_path}")
            return

        source = self._resolve_source()
        save_target = self.save_path.get().strip() or None

        self.settings["weights"] = str(weights_path)
        save_settings(self.settings)

        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._run_pipeline,
            args=(source, weights_path, save_target),
            daemon=True,
        )
        self.worker_thread.start()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_text.set("Running...")
        self.root.after(UI_REFRESH_MS, self._refresh_preview)

    def _run_pipeline(self, source, weights_path, save_target):
        try:
            run(
                source,
                weights_path,
                save_video=save_target,
                stop_event=self.stop_event,
                frame_callback=self._enqueue_frame,
            )
        except Exception as error:
            message = str(error)
            self.root.after(
                0,
                lambda text=message: messagebox.showerror("Pipeline error", text),
            )

    def _on_stop_pressed(self):
        if self.stop_event:
            self.stop_event.set()
        self.status_text.set("Stopping...")

    def _enqueue_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _refresh_preview(self):
        latest_frame = None
        try:
            while True:
                latest_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame is not None:
            self._draw_on_canvas(latest_frame)

        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(UI_REFRESH_MS, self._refresh_preview)
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_text.set("Ready")

    def _draw_on_canvas(self, frame):
        frame_height, frame_width = frame.shape[:2]
        scale = min(CANVAS_WIDTH / frame_width, CANVAS_HEIGHT / frame_height)
        scaled_width = int(frame_width * scale)
        scaled_height = int(frame_height * scale)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resample = Image.LANCZOS if scale < 1 else Image.BILINEAR
        pil_image = Image.fromarray(rgb_frame).resize(
            (scaled_width, scaled_height), resample
        )
        self.current_photo = ImageTk.PhotoImage(pil_image)

        self.preview.itemconfigure(self.preview_image_id, image=self.current_photo)
        self.preview.coords(
            self.preview_image_id,
            (CANVAS_WIDTH - scaled_width) // 2,
            (CANVAS_HEIGHT - scaled_height) // 2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli", help="Run headless on this source (path or camera index)"
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--save-video")
    args = parser.parse_args()

    if args.cli is not None:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            parser.error(f"Weights not found: {weights_path}")
        run(args.cli, weights_path, save_video=args.save_video)
        return

    root_window = tk.Tk()
    MonitorGUI(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
