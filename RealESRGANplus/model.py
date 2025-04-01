# Import the necessary libraries
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from os import path as osp
from tqdm import tqdm
import glob
import cv2
import os

# Import the local libraries
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from _utils import Reader, Writer

class RealESRGANplus:
    def __init__(self, model_name='RealESRGAN_x4plus', model_path=None, gpu_id=None, denoise_strength=0.5, outscale=4, tile=0, tile_pad=10, pre_pad=0, face_enhance=False, fp32=False, alpha_upsampler='realesrgan'):
        """
        Initialize the Real-ESRGAN model with given parameters.

        Args:
            model_name (str): Name of the ESRGAN model to use: [RealESRGAN_x4plus, RealESRNet_x4plus, RealESRGAN_x4plus_anime_6B, RealESRGAN_x2plus, realesr-animevideov3]
            model_path (str, optional): Path to the model weights.
            gpu_id (int, optional): GPU device ID.
            denoise_strength (float, optional): Strength of denoising (0-1).
            outscale (float, optional): Upscaling factor.
            tile (int, optional): Tile size for inference (0 means no tiling).
            tile_pad (int, optional): Padding size for tiles.
            pre_pad (int, optional): Padding before processing.
            face_enhance (bool, optional): Whether to enhance faces using GFPGAN (Resnet50 is used).
            fp32 (bool, optional): Use FP32 precision instead of FP16.
            alpha_upsampler (str, optional): The upsampler for the alpha channels. Options: realesrgan | bicubic.
        """
        
        # Set the model name, path, GPU ID, denoise strength, output scale, tile size, tile padding, pre padding, face enhancer, fp32, and alpha upsampler
        self.model_name = model_name.split('.')[0]
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.face_enhance = face_enhance
        self.fp32 = fp32
        self.alpha_upsampler = alpha_upsampler
        
        # Load the Real-ESRGAN model
        self.load_model()

    def load_model(self):
        # Define the model configurations
        model_configs = {
            'RealESRGAN_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'netscale': 4,
                'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            },
            'RealESRNet_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'netscale': 4,
                'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
            },
            'RealESRGAN_x4plus_anime_6B': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                'netscale': 4,
                'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            },
            'RealESRGAN_x2plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                'netscale': 2,
                'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
            },
            'realesr-animevideov3': {
                'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
                'netscale': 4,
                'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            }
        }

        # Get model configration based on the model name
        if self.model_name in model_configs:
            config = model_configs[self.model_name]
            self.model = config['model']
            self.netscale = config['netscale']
            file_url = config['file_url']
        else:
            raise ValueError(f'Unknown model name: {self.model_name}')

        # Download the model if not found locally
        if self.model_path is None:
            self.model_path = os.path.join('weights', f'{self.model_name}.pth')
            if not os.path.isfile(self.model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    self.model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None
                    )
                print(f'Model {self.model_name} loaded successfully from {self.model_path}.')
        print(f'{self.model_name} loading from {self.model_path}.')

        # load the Real-ESRGAN model
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            dni_weight=None,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id)        
        print(f'Model {self.model_name} loaded successfully.')

        # load face enhancer from GFPGAN
        if self.face_enhance:
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)
            print('Face enhancer loaded successfully.')

    def upscale_image(self, img):
        """
        Upscale the input image using the Real-ESRGAN model.
        Args:
        img: numpy image    
        Returns:
            results (list): List of upscaled images.
        """
        # Check if the input path is a file or a directory
        print('Processing image')
            
        try:
            if self.face_enhance and self.face_enhancer:
                _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self.upsampler.enhance(img, outscale=self.outscale)
        except RuntimeError as error:
            print('Error:', error)
        else:
            return output
    
    def upscale_video(self, input_path, output_path, ffmpeg_bin='ffmpeg'):
        """
        Upscale the input video using the Real-ESRGAN model.
        Args:
            input_path (str): Path to the input video.
            output_path (str): Path to save the upscaled video.
            ffmpeg_bin (str): Path to the ffmpeg binary.
        """
        os.makedirs(output_path, exist_ok=True)
        print(f'Folder created at {output_path}')

        # Convert the video to MP4 if it is a FLV file
        if input_path.endswith('.flv'):
            mp4_path = input_path.replace('.flv', '.mp4')
            os.system(f'ffmpeg -i {input_path} -codec copy {mp4_path}')
            input_path = mp4_path

        # Extract the video name
        video_name = osp.splitext(os.path.basename(input_path))[0]

        # Process the video
        return self.infer_video(input_path, video_name, output_path, ffmpeg_bin)
    
    def infer_video(self, input_path, video_name, output_path, ffmpeg_bin='ffmpeg'):
        """
        Upscale the input video using the Real-ESRGAN model.
        Args:
            input_path (str): Path to the input video.
            video_name (str): Name of the input video.
            output_path (str): Path to save the upscaled video.
            ffmpeg_bin (str): Path to the ffmpeg binary.
        """
        video_save_path = osp.join(output_path, f'{video_name}_out.mp4')

        # Check if the model supports face enhancement
        if 'anime' in self.model_name and self.face_enhance:
            print("Resnet50 don't not supported anime faces for detection. So, we turned this option off for you.")
            self.face_enhance = False
            self.face_enhancer = None
        
        # Initialize the reader and writer
        reader = Reader(input_path, video_name, output_path, ffmpeg_bin)
        audio = reader.get_audio()
        height, width = reader.get_resolution()
        fps = reader.get_fps()
        writer = Writer(self.outscale, video_save_path, fps, width, height)
        print("Video Reading and Writing Initialized")

        # Process the video
        print('Start video inference...')
        pbar = tqdm(total=len(reader), unit='frame', desc='inference')
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                if self.face_enhance and self.face_enhancer:
                    _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if output is None:
                    print('Error: Failed to process the video.')
                    break
                writer.write_frame(output)
            pbar.update(1)

        # Adding audio to the file
        if reader.audio:
            os.system(f'ffmpeg -i "{input_path}" -q:a 0 -map a "{reader.tmp_frames_folder}/{audio.aac}" -y')
            os.system(f'ffmpeg -i "{video_save_path}" -i "{reader.tmp_frames_folder}/{audio.aac}" -c:v copy -c:a aac -strict experimental "{video_save_path.replace("_out.mp4", "_audio.mp4")}" -y')
            os.remove(video_save_path)
            os.rename(video_save_path.replace("_out.mp4", "_audio.mp4"), video_save_path)

        reader.close()
        writer.close()

        return video_save_path
    

# Example usage
if __name__ == '__main__':
    # How to use the Image Upscale Model
    print('Real-ESRGAN Image Upscaler')
    upscaler = RealESRGANplus(model_name='RealESRGAN_x4plus',
                              model_path=None,
                              gpu_id=0,
                              denoise_strength=0.0,
                              outscale=4,
                              tile=0,
                              tile_pad=10,
                              pre_pad=0,
                              face_enhance=True,
                              fp32=True,
                              alpha_upsampler='realesrgan')
    
    img = cv2.imread('inputs\lr_image.png', cv2.IMREAD_UNCHANGED)
    result = upscaler.upscale_image(img)
    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite(f'outputs/out__image.png', result)

    # How to use the Video Upscale Model
    print('Real-ESRGAN Video Upscaler')
    upscaler = RealESRGANplus(model_name='RealESRGAN_x4plus',
                              model_path=None,
                              gpu_id=0,
                              denoise_strength=0.0,
                              outscale=4,
                              tile=0,
                              tile_pad=10,
                              pre_pad=0,
                              face_enhance=True,
                              fp32=True,
                              alpha_upsampler='bicubic')
    
    out_vid = upscaler.upscale_video('inputs\lr_video.mp4', 'outputs', ffmpeg_bin='ffmpeg')
    print(f'Upscaled video saved at: {out_vid}')