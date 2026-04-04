import logging
import shlex
import subprocess
import time
from collections.abc import Sequence

from .config import OverlayConfig, RenderVideoConfig, settings
from .reframing import CropPlan


logger = logging.getLogger(__name__)


def _format_cmd(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def run_cmd(args: Sequence[str], stage: str) -> None:
    started_at = time.perf_counter()
    logger.info("Running stage '%s': %s", stage, _format_cmd(args))
    process = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.perf_counter() - started_at
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed at stage '{stage}' after {elapsed:.2f}s:\n"
            f"{_format_cmd(args)}\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"
        )
    logger.info("Completed stage '%s' in %.2fs", stage, elapsed)


def download_video(youtube_url: str, output_path: str) -> None:
    run_cmd(
        [
            "yt-dlp",
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "-o",
            output_path,
            youtube_url,
        ],
        stage="youtube_download",
    )


def _video_encoding_args(render_config: RenderVideoConfig) -> list[str]:
    return [
        "-c:v",
        render_config.codec,
        "-preset",
        render_config.preset,
        "-crf",
        str(render_config.crf),
        "-pix_fmt",
        render_config.pixel_format,
        "-c:a",
        render_config.audio_codec,
        "-b:a",
        render_config.audio_bitrate,
        "-movflags",
        render_config.movflags,
    ]


def get_media_duration_seconds(path: str) -> float | None:
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            return None
        line = (process.stdout or "").strip()
        if not line:
            return None
        return float(line)
    except (OSError, ValueError):
        return None


def trim_video(
    input_path: str,
    output_path: str,
    start_time: str,
    end_time: str,
    render_config: RenderVideoConfig | None = None,
) -> None:
    active_render_config = render_config or settings.render_video
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ss",
            start_time,
            "-to",
            end_time,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            *(_video_encoding_args(active_render_config)),
            output_path,
        ],
        stage="trim_clip",
    )


def detect_scene_cuts(input_path: str, threshold: float = 0.35) -> list[float]:
    args = [
        "ffmpeg",
        "-i",
        input_path,
        "-filter:v",
        f"select='gt(scene,{threshold})',metadata=print",
        "-an",
        "-f",
        "null",
        "-",
    ]
    logger.info("Running stage 'scene_detect': %s", _format_cmd(args))
    process = subprocess.run(args, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed at stage 'scene_detect':\n{_format_cmd(args)}\n\nSTDERR:\n{process.stderr}"
        )
    cut_times: list[float] = []
    for line in (process.stderr or "").splitlines():
        if "pts_time:" not in line:
            continue
        try:
            value = line.split("pts_time:")[1].split()[0]
            cut_times.append(float(value))
        except (IndexError, ValueError):
            continue
    unique_cuts = sorted(set(cut_times))
    logger.info("Completed stage 'scene_detect' with %s cuts", len(unique_cuts))
    return unique_cuts


def _build_crop_coordinate_expression(
    values: list[tuple[float, int]],
    default_value: int,
) -> str:
    if not values:
        return str(default_value)
    if len(values) == 1:
        return str(values[0][1])

    expression = str(values[-1][1])
    for index in range(len(values) - 2, -1, -1):
        start_time, start_value = values[index]
        end_time, end_value = values[index + 1]
        if abs(end_time - start_time) < 1e-6:
            expression = str(end_value)
            continue
        expression = (
            f"if(lt(t,{start_time:.3f}),{start_value},"
            f"if(lt(t,{end_time:.3f}),{start_value}+({end_value - start_value})*"
            f"(t-{start_time:.3f})/{(end_time - start_time):.3f},{expression}))"
        )
    return expression


def build_crop_filter(
    crop_plan: CropPlan | None,
    render_config: RenderVideoConfig,
) -> str:
    if crop_plan is None:
        return (
            f"scale={render_config.width}:{render_config.height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={render_config.width}:{render_config.height},setsar=1"
        )

    x_values = [(keyframe.timestamp, keyframe.x) for keyframe in crop_plan.keyframes]
    y_values = [(keyframe.timestamp, keyframe.y) for keyframe in crop_plan.keyframes]
    x_expr = _build_crop_coordinate_expression(x_values, crop_plan.default_x)
    y_expr = _build_crop_coordinate_expression(y_values, crop_plan.default_y)
    return (
        f"crop={crop_plan.crop_width}:{crop_plan.crop_height}:'{x_expr}':'{y_expr}',"
        f"scale={render_config.width}:{render_config.height},setsar=1"
    )


def _build_canvas_filter(
    render_config: RenderVideoConfig,
    crop_plan: CropPlan | None = None,
) -> tuple[list[str], str]:
    if render_config.mode == "contain_blur_bg":
        return (
            [
                "[0:v]split=2[bg][fg]",
                (
                    f"[bg]scale={render_config.width}:{render_config.height}:"
                    "force_original_aspect_ratio=increase,"
                    f"crop={render_config.width}:{render_config.height},boxblur=20:10[bgfill]"
                ),
                (
                    f"[fg]scale={render_config.width}:{render_config.height}:"
                    "force_original_aspect_ratio=decrease[fgfit]"
                ),
                "[bgfill][fgfit]overlay=(W-w)/2:(H-h)/2,setsar=1[canvas]",
            ],
            "[canvas]",
        )

    return (
        [
            (
                f"[0:v]{build_crop_filter(crop_plan, render_config)}[canvas]"
            )
        ],
        "[canvas]",
    )


def get_render_transform_mode(render_mode: str | None = None) -> str:
    selected_mode = render_mode or settings.render_video.mode
    if selected_mode not in {"cover_crop", "contain_blur_bg"}:
        raise RuntimeError(f"Unsupported render mode: {selected_mode}")
    return selected_mode


def _escape_filter_path(path: str) -> str:
    escaped = path.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def build_filtergraph(
    ass_path: str | None,
    cta_path: str | None,
    watermark_path: str | None,
    render_config: RenderVideoConfig | None = None,
    overlay_config: OverlayConfig | None = None,
    crop_plan: CropPlan | None = None,
) -> tuple[list[str], str, str]:
    active_render_config = render_config or settings.render_video
    active_overlay_config = overlay_config or settings.render_overlay

    filter_parts, current_label = _build_canvas_filter(active_render_config, crop_plan=crop_plan)
    input_args: list[str] = []
    input_index = 1

    if cta_path:
        input_args.extend(["-loop", "1", "-i", cta_path])
        filter_parts.append(
            f"[{input_index}:v]scale={active_overlay_config.cta_width}:-1[cta]"
        )
        filter_parts.append(
            f"{current_label}[cta]overlay={active_overlay_config.cta_margin}:"
            f"{active_overlay_config.cta_margin}[with_cta]"
        )
        current_label = "[with_cta]"
        input_index += 1

    if watermark_path:
        input_args.extend(["-loop", "1", "-i", watermark_path])
        wm = active_overlay_config.watermark_size
        filter_parts.append(f"[{input_index}:v]scale={wm}:-1[watermark]")
        m = active_overlay_config.watermark_margin
        filter_parts.append(
            f"{current_label}[watermark]overlay=W-w-{m}:{m}[with_watermark]"
        )
        current_label = "[with_watermark]"
        input_index += 1

    if ass_path:
        ass_opt = f"'{_escape_filter_path(ass_path)}':charenc=UTF-8"
        filter_parts.append(f"{current_label}subtitles={ass_opt}[finalv]")
        current_label = "[finalv]"

    return input_args, ";".join(filter_parts), current_label


def burn_subtitles_and_overlays(
    input_path: str,
    output_path: str,
    ass_path: str | None,
    cta_path: str | None,
    watermark_path: str | None,
    render_mode: str | None = None,
    crop_plan: CropPlan | None = None,
) -> None:
    render_config = settings.render_video
    active_render_config = RenderVideoConfig(
        width=render_config.width,
        height=render_config.height,
        mode=get_render_transform_mode(render_mode),
        codec=render_config.codec,
        preset=render_config.preset,
        crf=render_config.crf,
        pixel_format=render_config.pixel_format,
        audio_codec=render_config.audio_codec,
        audio_bitrate=render_config.audio_bitrate,
        movflags=render_config.movflags,
    )
    extra_inputs, filter_complex, final_video_label = build_filtergraph(
        ass_path=ass_path,
        cta_path=cta_path,
        watermark_path=watermark_path,
        render_config=active_render_config,
        crop_plan=crop_plan if active_render_config.mode == "cover_crop" else None,
    )
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            *extra_inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            final_video_label,
            "-map",
            "0:a?",
            *(_video_encoding_args(active_render_config)),
            "-shortest",
            output_path,
        ],
        stage="render_final",
    )
