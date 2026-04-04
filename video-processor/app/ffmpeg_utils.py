import shlex
import subprocess

def run_cmd(cmd: str) -> None:
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")

def run_cmd_output(cmd: str) -> str:
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\n\nSTDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
    return process.stdout.strip()

def download_video(youtube_url: str, output_path: str) -> None:
    cmd = f'yt-dlp -f "bv*+ba/b" --merge-output-format mp4 -o {shlex.quote(output_path)} {shlex.quote(youtube_url)}'
    run_cmd(cmd)

def trim_video(input_path: str, output_path: str, start_time: str, end_time: str) -> None:
    cmd = f"ffmpeg -y -i {shlex.quote(input_path)} -ss {shlex.quote(start_time)} -to {shlex.quote(end_time)} -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 192k {shlex.quote(output_path)}"
    run_cmd(cmd)

def detect_scene_cuts(input_path: str, threshold: float = 0.35):
    cmd = f'ffmpeg -i {shlex.quote(input_path)} -filter:v "select=\'gt(scene,{threshold})\',metadata=print" -an -f null -'
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stderr = process.stderr or ""
    cut_times = []
    for line in stderr.splitlines():
        if "pts_time:" in line:
            try:
                value = line.split("pts_time:")[1].split()[0]
                cut_times.append(float(value))
            except Exception:
                continue
    return sorted(set(cut_times))

def burn_subtitles_and_overlays(input_path: str, output_path: str, srt_path: str | None, cta_path: str | None, watermark_path: str | None) -> None:
    filter_parts = []
    input_args = [f"-i {shlex.quote(input_path)}"]
    overlay_stage = "[base]"
    filter_parts.append("[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[base]")
    input_index = 1

    if cta_path:
        input_args.append(f"-i {shlex.quote(cta_path)}")
        filter_parts.append(f"{overlay_stage}[{input_index}:v]overlay=(W-w)/2:120[cta1]")
        overlay_stage = "[cta1]"
        input_index += 1

    if watermark_path:
        input_args.append(f"-i {shlex.quote(watermark_path)}")
        filter_parts.append(f"{overlay_stage}[{input_index}:v]overlay=W-w-30:H-h-30[wm1]")
        overlay_stage = "[wm1]"
        input_index += 1

    final_video_label = overlay_stage
    if srt_path:
        escaped_srt = srt_path.replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(
            f"{overlay_stage}subtitles='{escaped_srt}':force_style='FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=100'[finalv]"
        )
        final_video_label = "[finalv]"

    filter_complex = ";".join(filter_parts)
    cmd = f'ffmpeg -y {" ".join(input_args)} -filter_complex "{filter_complex}" -map "{final_video_label}" -map 0:a? -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 192k -shortest {shlex.quote(output_path)}'
    run_cmd(cmd)
