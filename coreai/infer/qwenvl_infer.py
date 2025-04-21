import json
from transformers import (
    AutoTokenizer,
    AutoProcessor,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    pass
try:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLModel
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as e:
    pass
from loguru import logger
from transformers import TextStreamer

import os
import torch
import glob
import natsort
from coreai.utils.symbols import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)
from coreai.utils.process_utils import replace_image_tokens
from .vl_infer_base import VLBase


class Qwen2_5_VL(VLBase):
    def __init__(self, model_path=None, processor_path=None, device="auto"):
        super().__init__(model_path, processor_path, device)
        # default: Load the model on the available device(s)

    def load_model(self, model_path):
        if model_path is None:
            model_path = "checkpoints/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        model.to(self.device)
        logger.info(f"model loaded from: {model_path}")
        return model

    def load_processor(self, processor_path):
        if processor_path is None:
            processor_path = "checkpoints/Qwen2.5-VL-3B-Instruct"
        processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.encode(
                self.tokenizer.pad_token
            )
        return processor

    def get_msg(self, text, image=None, system_msg=None):
        if image is None:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        elif os.path.isdir(image):
            image = glob.glob(os.path.join(image, "*.png"))
            image = natsort.sorted(image)

        if isinstance(image, list):
            a = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
            a["content"].append(
                {
                    "type": "video",
                    "video": image,
                    "max_pixels": 360 * 420,
                }
            )
            # for im in image:
            #     a['content'].append( {
            #             "type": "image",
            #             "image": im,
            #         },)
            return a

        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }

    def construct_prompt_ids(self, prompt, images, system=None):
        if system is None:
            system = "You are a helpful assistant."

        # convert <image> image tokens to QwenVL's
        prompt = replace_image_tokens(prompt)
        system_message = (
            f"{DEFAULT_IM_START_TOKEN}system\n{system}{DEFAULT_IM_END_TOKEN}\n"
        )

        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{prompt}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        user_input = system_message + user_input

        if DEFAULT_IMAGE_TOKEN in user_input:
            inputs = self.processor(
                text=[user_input],
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
            )

        elif DEFAULT_VIDEO_TOKEN in user_input:
            logger.warning(f"infer on video directly not supported yet.")
            if "Qwen2.5" in self.model_id:
                inputs = self.processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
            else:
                inputs = self.processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    return_tensors="pt",
                )
            prompt_input_ids = inputs["input_ids"]
        else:
            prompt_input_ids = self.processor.tokenizer(
                user_input,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]

        # return input_ids and pixel values
        return inputs

    def generate(
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
        """
        if image_special_token in prompt:
            assert prompt.count(image_special_token) == len(
                images if isinstance(images, list) else [images]
            ), f"{image_special_token} must have same num as images."
            inputs = self.construct_prompt_ids(prompt, images, system_msg)
        else:
            msg = self.get_msg(prompt, images)
            messages = [msg]
            if system_msg:
                messages.append({"role": "system", "content": system_msg})

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            print(image_inputs, video_inputs)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        if stream:
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        else:
            streamer = None

        generated_ids = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=500, streamer=streamer
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print(output_text)
        return output_text
