from dataclasses import dataclass, field

import cv2
import numpy as np
from config import (
    BBOX_MOVEMENT_THRESH,
    BRAKE_LATERAL_MARGIN,
    BRAKE_MIN_FOOT_RATIO,
    BRAKE_MIN_HEIGHT_RATIO,
    CLASS_CONSISTENCY_THRESH,
    CLASS_NAMES,
    CONFIDENCE_DROP_THRESH,
    DECISION_COLORS,
    HIGH_CONFIDENCE,
    HYSTERESIS_FRAMES,
    IOU_THRESH,
    MAX_LOST_FRAMES,
    STATIONARY_VELOCITY,
    WINDOW_SIZE,
)


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    class_id: int
    confidence: float
    class_history: list = field(default_factory=list)
    confidence_history: list = field(default_factory=list)
    bbox_history: list = field(default_factory=list)
    lost_frames: int = 0
    decision: str = "NORMAL"
    hysteresis_counter: int = 0
    pending_decision: str | None = None

    def __post_init__(self):
        self.class_history.append(self.class_id)
        self.confidence_history.append(self.confidence)
        self.bbox_history.append(_center(self.bbox))

    def update(self, bbox, cls, conf):
        self.bbox, self.class_id, self.confidence = bbox, cls, conf
        self.class_history.append(cls)
        self.confidence_history.append(conf)
        self.bbox_history.append(_center(bbox))
        self.lost_frames = 0


class Tracker:
    def __init__(self):
        self._next_id = 0
        self.tracks: list[Track] = []

    def update(self, detections):
        if not self.tracks:
            for d in detections:
                self._spawn(d)
            return self._active()

        n_t, n_d = len(self.tracks), len(detections)
        cost = np.zeros((n_t, n_d), dtype=np.float32)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                cost[i, j] = _iou(t.bbox, np.array(d["bbox"]))

        matched_t, matched_d = set(), set()
        while cost.size and cost.max() >= IOU_THRESH:
            i, j = np.unravel_index(np.argmax(cost), cost.shape)
            d = detections[j]
            self.tracks[i].update(
                np.array(d["bbox"], dtype=np.float32),
                d["class_id"],
                d["confidence"],
            )
            matched_t.add(i)
            matched_d.add(j)
            cost[i, :] = 0
            cost[:, j] = 0

        for i in range(n_t):
            if i not in matched_t:
                self.tracks[i].lost_frames += 1
        for j in range(n_d):
            if j not in matched_d:
                self._spawn(detections[j])

        self.tracks = [t for t in self.tracks if t.lost_frames <= MAX_LOST_FRAMES]
        return self._active()

    def _spawn(self, d):
        self.tracks.append(
            Track(
                track_id=self._next_id,
                bbox=np.array(d["bbox"], dtype=np.float32),
                class_id=d["class_id"],
                confidence=d["confidence"],
            )
        )
        self._next_id += 1

    def _active(self):
        return [t for t in self.tracks if t.lost_frames == 0]


def _stability(track):
    n = min(WINDOW_SIZE, len(track.class_history))
    classes = track.class_history[-n:]
    confs = np.array(track.confidence_history[-n:], dtype=np.float64)
    centers = track.bbox_history[-n:]

    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    dom = max(counts, key=counts.get)
    consistency = counts[dom] / len(classes)

    sudden_drop = len(confs) >= 2 and bool(
        np.any(np.diff(confs) < -CONFIDENCE_DROP_THRESH)
    )
    if len(centers) >= 2:
        pts = np.array(centers)
        velocity = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
    else:
        velocity = 0.0

    if sudden_drop or velocity > BBOX_MOVEMENT_THRESH:
        verdict = "DEGRADED"
    elif consistency < CLASS_CONSISTENCY_THRESH:
        verdict = "FLICKERING"
    else:
        verdict = "STABLE"

    return dom, consistency, float(confs.mean()), velocity, verdict


def _raw(dom, consistency, conf_mean, velocity, verdict):
    if verdict == "DEGRADED":
        return "NORMAL"
    if verdict == "FLICKERING":
        return "WARN"

    stationary = velocity < STATIONARY_VELOCITY
    if dom == 1 and consistency >= CLASS_CONSISTENCY_THRESH:
        if stationary or conf_mean >= HIGH_CONFIDENCE:
            return "SUPPRESS"
        return "WARN"
    if dom == 0 and consistency >= CLASS_CONSISTENCY_THRESH:
        if conf_mean >= HIGH_CONFIDENCE:
            return "BRAKE"
        return "WARN" if stationary else "BRAKE"
    return "NORMAL"


def _is_close(track, frame_h):
    if not frame_h:
        return True
    bbox_h = float(track.bbox[3] - track.bbox[1])
    bbox_bottom = float(track.bbox[3])
    return (
        bbox_h >= BRAKE_MIN_HEIGHT_RATIO * frame_h
        and bbox_bottom >= BRAKE_MIN_FOOT_RATIO * frame_h
    )


def _is_in_front(track, frame_w):
    if not frame_w:
        return True
    foot_x = (float(track.bbox[0]) + float(track.bbox[2])) / 2
    left_limit = BRAKE_LATERAL_MARGIN * frame_w
    right_limit = (1 - BRAKE_LATERAL_MARGIN) * frame_w
    return left_limit <= foot_x <= right_limit


def decide(track, frame_h=None, frame_w=None):
    raw = _raw(*_stability(track))

    if raw == "BRAKE":
        close = _is_close(track, frame_h)
        in_front = _is_in_front(track, frame_w)
        if not close and not in_front:
            raw = "NORMAL"
        elif not close or not in_front:
            raw = "WARN"

    if raw == track.decision:
        track.hysteresis_counter = 0
        track.pending_decision = None
        return track.decision

    # commit immediately if escalating to BRAKE, or if this is the track's
    # first non-default decision (otherwise still-image input would stay NORMAL)
    if raw == "BRAKE" or track.decision == "NORMAL":
        track.decision = raw
        track.hysteresis_counter = 0
        track.pending_decision = None
        return raw

    if track.pending_decision == raw:
        track.hysteresis_counter += 1
    else:
        track.pending_decision = raw
        track.hysteresis_counter = 1

    if track.hysteresis_counter >= HYSTERESIS_FRAMES:
        track.decision = raw
        track.hysteresis_counter = 0
        track.pending_decision = None

    return track.decision


_FONT = cv2.FONT_HERSHEY_DUPLEX
_LABEL_SCALE = 0.9
_LABEL_THICKNESS = 2
_LABEL_OUTLINE = 5
_STATUS_SCALE = 0.75
_STATUS_THICKNESS = 2
_BANNER_SCALE = 2.0
_BANNER_THICKNESS = 4
_DECISION_PRIORITY = {"BRAKE": 3, "WARN": 2, "SUPPRESS": 1, "NORMAL": 0}


def _draw_text(img, text, origin, color, scale, thickness):
    cv2.putText(img, text, origin, _FONT, scale, (0, 0, 0),
                thickness + _LABEL_OUTLINE - _LABEL_THICKNESS, cv2.LINE_AA)
    cv2.putText(img, text, origin, _FONT, scale, color,
                thickness, cv2.LINE_AA)


def _aggregate_decision(tracks):
    if not tracks:
        return None
    settled = [t for t in tracks if len(t.class_history) >= HYSTERESIS_FRAMES]
    if not settled:
        return None
    return max(settled, key=lambda t: _DECISION_PRIORITY[t.decision]).decision


def _draw_banner(img, text, color):
    h, w = img.shape[:2]
    (tw, th), _baseline = cv2.getTextSize(
        text, _FONT, _BANNER_SCALE, _BANNER_THICKNESS
    )
    pad_x, pad_y = 24, 14
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    box_x = (w - box_w) // 2
    box_y = h - box_h - 16
    overlay = img.copy()
    cv2.rectangle(
        overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), color, -1
    )
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(
        img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 2
    )
    _draw_text(
        img,
        text,
        (box_x + pad_x, box_y + box_h - pad_y),
        (255, 255, 255),
        _BANNER_SCALE,
        _BANNER_THICKNESS,
    )


def draw(frame, tracks, frame_idx, fps):
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, 0), (w, 38), (25, 25, 25), -1)
    _draw_text(
        out,
        f"Frame {frame_idx}  FPS {fps:.1f}  Tracks {len(tracks)}",
        (12, 27),
        (240, 240, 240),
        _STATUS_SCALE,
        _STATUS_THICKNESS,
    )

    for t in tracks:
        color = DECISION_COLORS.get(t.decision, (160, 160, 160))
        x1, y1, x2, y2 = t.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label = f"#{t.track_id} {CLASS_NAMES.get(t.class_id, '?')} {t.confidence:.2f} [{t.decision}]"
        (tw, th), _baseline = cv2.getTextSize(label, _FONT, _LABEL_SCALE, _LABEL_THICKNESS)
        pad_x, pad_y = 8, 6
        tag_top = max(0, y1 - th - 2 * pad_y)
        cv2.rectangle(out, (x1, tag_top), (x1 + tw + 2 * pad_x, y1), color, -1)
        _draw_text(
            out, label,
            (x1 + pad_x, y1 - pad_y),
            (255, 255, 255),
            _LABEL_SCALE,
            _LABEL_THICKNESS,
        )

        for i, c in enumerate(t.class_history[-WINDOW_SIZE:]):
            cell_color = (0, 0, 200) if c == 0 else (0, 180, 0)
            cx = x1 + i * 12
            cv2.rectangle(out, (cx, y2 + 6), (cx + 11, y2 + 14), cell_color, -1)

    aggregate = _aggregate_decision(tracks)
    if aggregate:
        banner_color = DECISION_COLORS.get(aggregate, (160, 160, 160))
        _draw_banner(out, aggregate, banner_color)

    return out
