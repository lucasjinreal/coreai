# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_AUDIOTOKEN = "<audio>"

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_VIDEO_TOKEN = "<video>"


DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN_QWENVL = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN_QWENVL = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

QWENVL_IMAGE_TOKEN = f'{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN_QWENVL}{VISION_END_TOKEN}'

SYSTEM_MESSAGE = "You are a helpful assistant."
SYSTEM_NAV = """Imagine you are a robot programmed for navigation 
tasks. You have been given a video of historical 
observations and a image of current observation. 
Analyze this series of images to decide your next 
move, which could involve turning left or right by a 
specific degree and then moving forward a certain distance. If arrived target, should stop.

The historical images in <HIS></HIS>, current observation in <OBS></OBS>, you navigation task is after <NAV>.
Your response should be in: The next action is turn left xx°, then move foward xx cm.
If you judge arrived target, then response: The next action is stop."""
