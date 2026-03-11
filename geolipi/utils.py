
from PIL import Image
from typing import List, Literal

def frames_to_animation(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 10,
    format: Literal['gif', 'webp', 'mp4'] = 'gif',
    mp4_quality: Literal['lossless', 'high', 'medium'] = 'high',
) -> str:
    """
    Takes a sequence of PIL Image frames and saves them as an animated GIF, WebP, or MP4.

    Parameters:
    - frames (List[Image.Image]): List of PIL Image frames.
    - output_path (str): Path where the animation file will be saved.
    - fps (int): Frames per second for the animation. Default is 10.
    - format (str): Output format: 'gif', 'webp', or 'mp4'. Default is 'gif'.
    - mp4_quality (str): Quality for MP4 encoding. Only used when format='mp4'.
        - 'lossless': No compression (CRF 0, very large files)
        - 'high': Visually lossless (CRF 17, recommended for quality)
        - 'medium': Good quality with smaller file size (CRF 23)

    Returns:
    - str: The path to the saved animation file.
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Ensure all frames have the same size (resize to first frame's size)
    base_size = frames[0].size
    frames = [f.resize(base_size, Image.Resampling.LANCZOS) if f.size != base_size else f 
              for f in frames]
    
    # Calculate duration in milliseconds
    duration_ms = int(1000 / fps)
    
    # Ensure correct file extension
    if not output_path.lower().endswith(f'.{format}'):
        output_path = f"{output_path}.{format}"
    
    # Save animation
    if format == 'gif':
        # Convert RGBA to P mode for better GIF support
        frames_p = [f.convert('P', palette=Image.Palette.ADAPTIVE, colors=256) for f in frames]
        frames_p[0].save(
            output_path,
            save_all=True,
            append_images=frames_p[1:],
            duration=duration_ms,
            loop=0,  # 0 = infinite loop
            optimize=True
        )
    elif format == 'webp':
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            lossless=True
        )
    elif format == 'mp4':
        import imageio.v2 as iio
        import numpy as np

        crf_map = {"lossless": 0, "high": 17, "medium": 23}
        crf = crf_map.get(mp4_quality, 17)

        frame_arrays = [np.asarray(f.convert("RGB")) for f in frames]

        codec = "libx264"
        if mp4_quality == "lossless":
            output_params = ["-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"]
            pixelformat = "yuv444p"
        else:
            output_params = ["-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2"]
            pixelformat = "yuv420p"

        w = iio.get_writer(
            output_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec=codec,
            output_params=output_params,
            pixelformat=pixelformat,
        )
        for fr in frame_arrays:
            w.append_data(fr)
        w.close()
    
    return output_path
