from os import path as osp
import numpy as np
import shutil
import glob
import cv2
import os

# Import ffmpeg
import ffmpeg

# Important functions for video-streaming data handling
# All of these function are from the https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

class Reader:
    def __init__(self, input_path, video_name, output_path, ffmpeg_bin):
        self.audio = None
        self.input_fps = None
        self.width = None
        self.height = None
        self.nb_frames = None
        self.idx = 0
        self.paths = []
        self.process_video(input_path, video_name, output_path, ffmpeg_bin)

    def process_video(self, video_path, video_name, output_path, ffmpeg_bin):
        """Extracts metadata and frames from the video."""
        # Get video metadata using ffmpeg
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        # Extract metadata
        if video_stream:
            self.width = int(video_stream['width'])
            self.height = int(video_stream['height'])
            self.input_fps = eval(video_stream['r_frame_rate'])  # Convert "30/1" to float(30)
            self.nb_frames = int(video_stream.get('nb_frames', 0))  # Some formats may not have nb_frames
        if audio_stream:
            self.audio = audio_stream['codec_name']  # Store audio codec name

        # Extract frames from the video
        self.tmp_frames_folder = osp.join(output_path, f'{video_name}_inp_tmp_frames')
        os.makedirs(self.tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {video_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {self.tmp_frames_folder}/frame%08d.png')
        print(f"Extracting frames to: {self.tmp_frames_folder}")

        # Get all extracted frame paths
        self.paths = sorted(glob.glob(osp.join(self.tmp_frames_folder, "frame*.png")))
        self.nb_frames = len(self.paths)
        print(f"Extracted {self.nb_frames} frames from video.")

    def get_resolution(self):
        return self.height, self.width
    
    def get_fps(self):
        if self.input_fps is not None:
            return self.input_fps
        return 24
    
    def get_audio(self):
        return self.audio

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        return self.get_frame_from_list()
    
    def __len__(self):
        return self.nb_frames

    def close(self):
        shutil.rmtree(self.tmp_frames_folder)
        pass

class Writer:
    def __init__(self, outscale, output_path, fps, width, height):
        """Initialize OpenCV VideoWriter (without audio)."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        self.output_path = output_path
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (outscale*width, outscale*height))

    def write_frame(self, frame):
        """Write a single frame to the output video."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")

        if frame.dtype != np.uint8:
            raise ValueError("Frame data type must be np.uint8.")

        self.video_writer.write(frame)

    def close(self):
        """Release the video writer."""
        self.video_writer.release()