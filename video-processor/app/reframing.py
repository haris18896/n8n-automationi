import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SubjectAwareCropConfig, settings


logger = logging.getLogger(__name__)

TARGET_ASPECT_RATIO = 9 / 16
FACE_METHOD = "face"
OBJECT_METHOD = "object"
MOTION_METHOD = "motion"
CENTER_METHOD = "center"


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


@dataclass(frozen=True)
class SampledFrame:
    timestamp: float
    frame_index: int
    frame_bgr: np.ndarray
    frame_gray: np.ndarray
    analysis_width: int
    analysis_height: int
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class FocusRegion:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    method: str


@dataclass(frozen=True)
class FocusPoint:
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    method: str


@dataclass(frozen=True)
class CropKeyframe:
    timestamp: float
    x: int
    y: int


@dataclass(frozen=True)
class CropPlan:
    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    keyframes: list[CropKeyframe]
    default_x: int
    default_y: int
    method_counts: dict[str, int]


class ReframingAnalyzer:
    def __init__(self, config: SubjectAwareCropConfig | None = None):
        self.config = config or settings.subject_aware_crop
        haar_dir = Path(cv2.data.haarcascades)
        self.face_cascades = self._load_cascades(
            [
                haar_dir / "haarcascade_frontalface_default.xml",
                haar_dir / "haarcascade_profileface.xml",
            ]
        )
        self.subject_cascades = self._load_cascades(
            [
                haar_dir / "haarcascade_upperbody.xml",
                haar_dir / "haarcascade_fullbody.xml",
                haar_dir / "haarcascade_lowerbody.xml",
            ]
        )
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _load_cascades(paths: list[Path]) -> list[cv2.CascadeClassifier]:
        cascades: list[cv2.CascadeClassifier] = []
        for path in paths:
            if not path.exists():
                continue
            cascade = cv2.CascadeClassifier(str(path))
            if cascade.empty():
                continue
            cascades.append(cascade)
        return cascades


def get_video_metadata(video_path: str) -> VideoMetadata:
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for reframing: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 30.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions for reframing: {video_path}")
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def _resize_for_analysis(frame_bgr: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float, float]:
    height, width = frame_bgr.shape[:2]
    if max(height, width) <= max_dimension:
        return frame_bgr, 1.0, 1.0
    scale = max_dimension / float(max(height, width))
    resized = cv2.resize(
        frame_bgr,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, width / resized.shape[1], height / resized.shape[0]


def sample_video_frames(
    video_path: str,
    interval_seconds: float,
    max_dimension: int,
) -> tuple[VideoMetadata, list[SampledFrame]]:
    metadata = get_video_metadata(video_path)
    capture = cv2.VideoCapture(video_path)
    sampled_frames: list[SampledFrame] = []
    try:
        sample_stride = max(1, int(round(metadata.fps * max(interval_seconds, 0.1))))
        frame_indices = list(range(0, max(metadata.frame_count, 1), sample_stride))
        if metadata.frame_count > 0 and frame_indices[-1] != metadata.frame_count - 1:
            frame_indices.append(metadata.frame_count - 1)

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            analysis_frame, scale_x, scale_y = _resize_for_analysis(frame, max_dimension)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            sampled_frames.append(
                SampledFrame(
                    timestamp=frame_index / metadata.fps,
                    frame_index=frame_index,
                    frame_bgr=analysis_frame,
                    frame_gray=gray,
                    analysis_width=analysis_frame.shape[1],
                    analysis_height=analysis_frame.shape[0],
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            )
        return metadata, sampled_frames
    finally:
        capture.release()


def _scale_box_to_source(sampled_frame: SampledFrame, x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
    return (
        x * sampled_frame.scale_x,
        y * sampled_frame.scale_y,
        w * sampled_frame.scale_x,
        h * sampled_frame.scale_y,
    )


def _region_from_box(
    sampled_frame: SampledFrame,
    x: int,
    y: int,
    w: int,
    h: int,
    confidence: float,
    method: str,
) -> FocusRegion:
    sx, sy, sw, sh = _scale_box_to_source(sampled_frame, x, y, w, h)
    return FocusRegion(
        center_x=sx + sw / 2.0,
        center_y=sy + sh / 2.0,
        width=sw,
        height=sh,
        confidence=confidence,
        method=method,
    )


def _score_region(region: FocusRegion, frame_width: int, frame_height: int) -> float:
    area_ratio = (region.width * region.height) / float(frame_width * frame_height)
    center_distance = np.hypot(
        (region.center_x - frame_width / 2.0) / max(frame_width, 1),
        (region.center_y - frame_height / 2.0) / max(frame_height, 1),
    )
    center_bonus = max(0.0, 1.0 - center_distance)
    return (area_ratio * 4.0) + (region.confidence * 0.45) + (center_bonus * 0.25)


def detect_faces(
    sampled_frame: SampledFrame,
    metadata: VideoMetadata,
    analyzer: ReframingAnalyzer,
) -> list[FocusRegion]:
    if not analyzer.config.face_detection_enabled:
        return []

    min_area = metadata.width * metadata.height * analyzer.config.min_face_box_area_ratio
    candidates: list[FocusRegion] = []
    equalized = cv2.equalizeHist(sampled_frame.frame_gray)

    for cascade in analyzer.face_cascades:
        detections = cascade.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        for x, y, w, h in detections:
            if (w * sampled_frame.scale_x) * (h * sampled_frame.scale_y) < min_area:
                continue
            candidates.append(
                _region_from_box(
                    sampled_frame,
                    x,
                    y,
                    w,
                    h,
                    confidence=0.92,
                    method=FACE_METHOD,
                )
            )

        mirrored = cv2.flip(equalized, 1)
        detections = cascade.detectMultiScale(
            mirrored,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        for x, y, w, h in detections:
            original_x = sampled_frame.analysis_width - x - w
            if (w * sampled_frame.scale_x) * (h * sampled_frame.scale_y) < min_area:
                continue
            candidates.append(
                _region_from_box(
                    sampled_frame,
                    original_x,
                    y,
                    w,
                    h,
                    confidence=0.88,
                    method=FACE_METHOD,
                )
            )

    candidates.sort(
        key=lambda region: _score_region(region, metadata.width, metadata.height),
        reverse=True,
    )
    return candidates


def _detect_people_with_hog(
    sampled_frame: SampledFrame,
    analyzer: ReframingAnalyzer,
) -> list[tuple[int, int, int, int, float]]:
    boxes, weights = analyzer.hog_detector.detectMultiScale(
        sampled_frame.frame_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    return [(x, y, w, h, float(weight)) for (x, y, w, h), weight in zip(boxes, weights)]


def detect_subjects(
    sampled_frame: SampledFrame,
    metadata: VideoMetadata,
    analyzer: ReframingAnalyzer,
) -> list[FocusRegion]:
    if not analyzer.config.object_detection_enabled:
        return []

    min_area = metadata.width * metadata.height * analyzer.config.min_subject_box_area_ratio
    candidates: list[FocusRegion] = []

    for x, y, w, h, weight in _detect_people_with_hog(sampled_frame, analyzer):
        source_area = (w * sampled_frame.scale_x) * (h * sampled_frame.scale_y)
        if source_area < min_area:
            continue
        confidence = min(0.95, 0.55 + max(0.0, weight) * 0.08)
        if confidence < analyzer.config.min_subject_confidence:
            continue
        candidates.append(
            _region_from_box(sampled_frame, x, y, w, h, confidence, OBJECT_METHOD)
        )

    for cascade in analyzer.subject_cascades:
        detections = cascade.detectMultiScale(
            sampled_frame.frame_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(32, 32),
        )
        for x, y, w, h in detections:
            source_area = (w * sampled_frame.scale_x) * (h * sampled_frame.scale_y)
            if source_area < min_area:
                continue
            candidates.append(
                _region_from_box(sampled_frame, x, y, w, h, 0.72, OBJECT_METHOD)
            )

    candidates.sort(
        key=lambda region: _score_region(region, metadata.width, metadata.height),
        reverse=True,
    )
    return candidates


def estimate_motion_focus(
    sampled_frame: SampledFrame,
    previous_frame_gray: np.ndarray | None,
    metadata: VideoMetadata,
    analyzer: ReframingAnalyzer,
) -> FocusRegion | None:
    if not analyzer.config.motion_fallback_enabled or previous_frame_gray is None:
        return None

    diff = cv2.absdiff(previous_frame_gray, sampled_frame.frame_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, motion_mask = cv2.threshold(
        diff,
        analyzer.config.motion_pixel_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    source_area = (w * sampled_frame.scale_x) * (h * sampled_frame.scale_y)
    min_area = metadata.width * metadata.height * analyzer.config.motion_area_ratio_threshold
    if source_area < min_area:
        return None

    mask_area = float(cv2.contourArea(contour))
    confidence = min(0.75, 0.35 + (mask_area / max(sampled_frame.analysis_width * sampled_frame.analysis_height, 1)))
    return _region_from_box(sampled_frame, x, y, w, h, confidence, MOTION_METHOD)


def choose_focus_region(
    sampled_frame: SampledFrame,
    metadata: VideoMetadata,
    analyzer: ReframingAnalyzer,
    previous_frame_gray: np.ndarray | None,
) -> FocusPoint:
    face_candidates = detect_faces(sampled_frame, metadata, analyzer)
    if face_candidates:
        best = face_candidates[0]
        return FocusPoint(sampled_frame.timestamp, best.center_x, best.center_y, best.confidence, best.method)

    subject_candidates = detect_subjects(sampled_frame, metadata, analyzer)
    if subject_candidates:
        best = subject_candidates[0]
        return FocusPoint(sampled_frame.timestamp, best.center_x, best.center_y, best.confidence, best.method)

    motion_region = estimate_motion_focus(
        sampled_frame,
        previous_frame_gray,
        metadata,
        analyzer,
    )
    if motion_region is not None:
        return FocusPoint(
            sampled_frame.timestamp,
            motion_region.center_x,
            motion_region.center_y,
            motion_region.confidence,
            motion_region.method,
        )

    return FocusPoint(
        timestamp=sampled_frame.timestamp,
        center_x=metadata.width / 2.0,
        center_y=metadata.height / 2.0,
        confidence=0.0,
        method=CENTER_METHOD,
    )


def smooth_focus_path(
    focus_points: list[FocusPoint],
    metadata: VideoMetadata,
    analyzer: ReframingAnalyzer,
) -> list[FocusPoint]:
    if not focus_points:
        return []

    alpha = min(max(analyzer.config.smoothing_alpha, 0.01), 1.0)
    deadzone = max(analyzer.config.deadzone_px, 0)

    smoothed_points: list[FocusPoint] = []
    prev_x = focus_points[0].center_x
    prev_y = focus_points[0].center_y

    for point in focus_points:
        next_x = point.center_x
        next_y = point.center_y

        if abs(next_x - prev_x) <= deadzone:
            next_x = prev_x
        else:
            next_x = (alpha * next_x) + ((1.0 - alpha) * prev_x)

        if abs(next_y - prev_y) <= deadzone:
            next_y = prev_y
        else:
            next_y = (alpha * next_y) + ((1.0 - alpha) * prev_y)

        prev_x = float(np.clip(next_x, 0.0, metadata.width))
        prev_y = float(np.clip(next_y, 0.0, metadata.height))
        smoothed_points.append(
            FocusPoint(
                timestamp=point.timestamp,
                center_x=prev_x,
                center_y=prev_y,
                confidence=point.confidence,
                method=point.method,
            )
        )
    return smoothed_points


def _target_crop_size(metadata: VideoMetadata) -> tuple[int, int]:
    source_ratio = metadata.width / float(metadata.height)
    if source_ratio >= TARGET_ASPECT_RATIO:
        crop_height = metadata.height
        crop_width = max(1, int(round(crop_height * TARGET_ASPECT_RATIO)))
    else:
        crop_width = metadata.width
        crop_height = max(1, int(round(crop_width / TARGET_ASPECT_RATIO)))
    crop_width = min(crop_width, metadata.width)
    crop_height = min(crop_height, metadata.height)
    return crop_width, crop_height


def _clamp_crop_origin(center: float, crop_size: int, frame_size: int) -> int:
    origin = int(round(center - crop_size / 2.0))
    return int(np.clip(origin, 0, max(frame_size - crop_size, 0)))


def _compress_keyframes(
    keyframes: list[CropKeyframe],
    analyzer: ReframingAnalyzer,
) -> list[CropKeyframe]:
    if not keyframes:
        return []

    reduced = [keyframes[0]]
    for keyframe in keyframes[1:-1]:
        previous = reduced[-1]
        moved_far_enough = (
            abs(keyframe.x - previous.x) > analyzer.config.deadzone_px
            or abs(keyframe.y - previous.y) > analyzer.config.deadzone_px
        )
        spaced_enough = (
            keyframe.timestamp - previous.timestamp
            >= analyzer.config.min_keyframe_spacing_seconds
        )
        if moved_far_enough and spaced_enough:
            reduced.append(keyframe)

    if len(keyframes) > 1:
        reduced.append(keyframes[-1])

    if len(reduced) <= analyzer.config.max_keyframes:
        return reduced

    selected_indices = np.linspace(
        0,
        len(reduced) - 1,
        analyzer.config.max_keyframes,
        dtype=int,
    )
    return [reduced[index] for index in selected_indices]


def generate_crop_plan(
    metadata: VideoMetadata,
    focus_points: list[FocusPoint],
    analyzer: ReframingAnalyzer,
) -> CropPlan:
    crop_width, crop_height = _target_crop_size(metadata)
    default_x = _clamp_crop_origin(metadata.width / 2.0, crop_width, metadata.width)
    default_y = _clamp_crop_origin(metadata.height / 2.0, crop_height, metadata.height)

    if not focus_points:
        return CropPlan(
            source_width=metadata.width,
            source_height=metadata.height,
            crop_width=crop_width,
            crop_height=crop_height,
            keyframes=[CropKeyframe(timestamp=0.0, x=default_x, y=default_y)],
            default_x=default_x,
            default_y=default_y,
            method_counts={CENTER_METHOD: 1},
        )

    keyframes: list[CropKeyframe] = []
    method_counts: dict[str, int] = {}
    for point in focus_points:
        method_counts[point.method] = method_counts.get(point.method, 0) + 1
        keyframes.append(
            CropKeyframe(
                timestamp=point.timestamp,
                x=_clamp_crop_origin(point.center_x, crop_width, metadata.width),
                y=_clamp_crop_origin(point.center_y, crop_height, metadata.height),
            )
        )

    compressed_keyframes = _compress_keyframes(keyframes, analyzer)
    return CropPlan(
        source_width=metadata.width,
        source_height=metadata.height,
        crop_width=crop_width,
        crop_height=crop_height,
        keyframes=compressed_keyframes,
        default_x=default_x,
        default_y=default_y,
        method_counts=method_counts,
    )


def analyze_subject_aware_crop(video_path: str) -> CropPlan:
    analyzer = ReframingAnalyzer()
    logger.info("Starting subject-aware crop analysis for %s", video_path)
    metadata, sampled_frames = sample_video_frames(
        video_path,
        interval_seconds=analyzer.config.analysis_interval_seconds,
        max_dimension=analyzer.config.analysis_max_dimension,
    )
    logger.info(
        "Crop analysis sampled %s frames over %.2fs",
        len(sampled_frames),
        metadata.duration,
    )
    if not sampled_frames:
        logger.warning("No frames sampled for %s; using center crop fallback", video_path)
        return generate_crop_plan(metadata, [], analyzer)

    previous_gray: np.ndarray | None = None
    focus_points: list[FocusPoint] = []
    method_counts: dict[str, int] = {}
    for sampled_frame in sampled_frames:
        focus = choose_focus_region(
            sampled_frame,
            metadata,
            analyzer,
            previous_gray,
        )
        method_counts[focus.method] = method_counts.get(focus.method, 0) + 1
        focus_points.append(focus)
        previous_gray = sampled_frame.frame_gray

    smoothed_points = smooth_focus_path(focus_points, metadata, analyzer)
    crop_plan = generate_crop_plan(metadata, smoothed_points, analyzer)
    logger.info(
        "Crop analysis summary frames=%s methods=%s keyframes=%s crop=%sx%s",
        len(sampled_frames),
        method_counts,
        len(crop_plan.keyframes),
        crop_plan.crop_width,
        crop_plan.crop_height,
    )
    return crop_plan
