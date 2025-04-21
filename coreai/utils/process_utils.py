from .symbols import LLAVA_VIDEO_TOKEN, LLAVA_IMAGE_TOKEN
from .symbols import VISION_START_TOKEN, VISION_END_TOKEN
import re
from .symbols import (
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_TOKEN_QWENVL,
    DEFAULT_VIDEO_TOKEN_QWENVL,
)


def replace_image_tokens_to_qwenvl_default(input_string, is_video=False):
    if is_video:
        if LLAVA_IMAGE_TOKEN in input_string:
            pattern = r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        else:
            pattern = r"\n?" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN_QWENVL + VISION_END_TOKEN
    else:
        pattern = r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN_QWENVL + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)
