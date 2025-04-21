from datetime import time
from typing import List, Dict, Union

import base64
import requests
from PIL import Image
from io import BytesIO

from coreai.utils.process_utils import replace_image_tokens_to_qwenvl_default
from qwen_vl_utils import process_vision_info, fetch_image

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
except ImportError as e:
    pass

from loguru import logger
import os
from coreai.utils.symbols import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)
import uuid


class QwenVLvLLMInfer:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        limit_mm_per_prompt: int = 16,
        modality="image",
    ):
        if not os.path.exists(model_path):
            logger.error(f'{model_path} not found.')
        self.engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True,  # 避免图编译问题
            disable_log_requests=True,
            trust_remote_code=True,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 480 * 28 * 28,
                "fps": 1,
            },
            limit_mm_per_prompt={"image": limit_mm_per_prompt},
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.sampling_params = SamplingParams(
            temperature=0.8, top_p=0.9, max_tokens=2048, skip_special_tokens=True
        )

        if modality == "image":
            self.placeholder = "<|image_pad|>"
        elif modality == "video":
            self.placeholder = "<|video_pad|>"

    def construct_prompt_templated(self, prompt, images, system=None):
        if system is None:
            system = "You are a helpful assistant."

        # convert <image> image tokens to QwenVL's
        # TODO: handle if video inputs
        prompt = replace_image_tokens_to_qwenvl_default(prompt)
        system_message = (
            f"{DEFAULT_IM_START_TOKEN}system\n{system}{DEFAULT_IM_END_TOKEN}\n"
        )

        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{prompt}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        user_input = system_message + user_input

        return user_input

    async def generate_single_turn(
        self,
        prompt,
        images,
        stream=True,
        max_size=700,
        verbose=False,
        prevent_more_image=True,
        system_msg=None,
        image_special_token="<image>",
    ):
        """
        judge if prompt contains <image> placeholder
        if does, then will insert images tokens to position of <image>

        currently only supports single prompt and mulitple images
        single turn maybe
        """
        if image_special_token in prompt:
            assert prompt.count(image_special_token) == len(
                images if isinstance(images, list) else [images]
            ), f"{image_special_token} must have same num as images."
            prompt_templated = self.construct_prompt_templated(
                prompt, images, system_msg
            )
        else:
            logger.error(f'{prompt} seems doesnt contains {image_special_token} but images were not empty.')
            raise NotImplementedError
        
        if verbose:
            print(f'prompt_templated {prompt_templated}')
            
        images_smart_resized = [
            fetch_image({"image": i, **self.engine_args.mm_processor_kwargs})
            for i in images
        ]
        outputs_generator = self.engine.generate(
            {
                "prompt": prompt_templated,
                "multi_modal_data": {"image": images_smart_resized},
            },
            self.sampling_params,
            str(uuid.uuid4()) 
        )
        generated_text = ""
        async for o in outputs_generator:
            generated_text = o.outputs[0].text
            # if verbose:
            #     print(generated_text)
        return generated_text
