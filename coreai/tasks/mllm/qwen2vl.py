from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
import torch
from .base import *

# qwen2 vl for API inferencing


class VisionQnA(VisionQnABase):
    format: str = "internal"
    model_name: str = "qwen2vl"
    vision_layers: List[str] = ["resampler", "vpm"]

    def __init__(
        self,
        model_id: str,
        device: str,
        device_map: str = "auto",
        extra_params={},
        format=None,
    ):
        super().__init__(model_id, device, device_map, extra_params, format)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self.params["torch_dtype"] = torch.float16
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            **self.params, trust_remote_code=True,
        ).eval()

        min_pixels = 256 * 28 * 28
        # max_pixels = 1280*28*28
        max_pixels = 512 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if not (
            extra_params.get("load_in_4bit", False)
            or extra_params.get("load_in_8bit", False)
        ):
            self.model = self.model.to(
                dtype=self.params["torch_dtype"], device=self.device
            )

        self.loaded_banner()

    async def stream_chat_with_images(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        # default uses num_beams: 3, but if streaming/sampling is requested, switch the defaults.
        default_params = {
            "num_beams": 1,
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": 0.8,
            "repetition_penalty": 1.3,
        }

        params = self.get_generation_params(request, default_params)

        del params["use_cache"]

        msgs, images = get_template_need_msgs(request.messages)
        if len(images) > 15:
            from qwen_vl_utils import process_vision_info

            print(f"video mode. {len(images)}")
            text_prompt = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True
            )

            for m in msgs:
                if m.role == "assistant":
                    for c in m.content:
                        if c.type == "image":
                            c["type"] = "video"

            image_inputs, video_inputs = process_vision_info(msgs)
            print(f"video_inputs:  {video_inputs[0].shape}")
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            text_prompt = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True
            )
            print(text_prompt)
            print(f"got images num: {len(images)}")

            inputs = self.processor(
                text=[text_prompt], images=images, padding=True, return_tensors="pt"
            )
            inputs.to(self.model.device)
            print(inputs["pixel_values"].shape)

        generation_kwargs = dict(**inputs, **params,)

        # for new_text in threaded_streaming_generator_bare(
        #     generate=self.model.generate,
        #     tokenizer=self.tokenizer,
        #     generation_kwargs=generation_kwargs,
        # ):
        #     end = new_text.find(self.eos_token)
        #     if end == -1:
        #         yield new_text
        #     else:
        #         yield new_text[:end]
        #         break

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                # streamer=streamer,
                use_cache=True,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            print(output_text)
            yield output_text[0]


def get_template_need_msgs(messages):
    result_msgs = []
    images = []
    if messages and messages[-1].role == "assistant":
        generation_msg += messages[-1].content[0].text
        messages.pop(-1)

    for m in messages:
        if m.role == "user":
            text = ""
            has_image = False

            one = {
                "role": "user",
                "content": [],
            }

            for c in m.content:
                if c.type == "image_url":
                    images.extend([url_to_image(c.image_url.url)])
                    has_image = True
                    one["content"].append(
                        {"type": "image",}
                    )
                if c.type == "text":
                    one["content"].append({"type": "text", "text": c.text})
            result_msgs.append(one)
        elif m.role == "assistant":
            one = {
                "role": "assistant",
                "content": [],
            }
            for c in m.content:
                if c.type == "text":
                    one["content"].append({"type": "text", "text": c.text})
            result_msgs.append(one)
        elif m.role == "system":
            one = {
                "role": "system",
                "content": [],
            }
            for c in m.content:
                if c.type == "text":
                    one["content"].append({"type": "text", "text": c.text})
            result_msgs.append(one)
    return result_msgs, images


def url_to_image(img_url: str) -> Image.Image:
    if img_url.startswith("http"):
        response = requests.get(img_url)

        img_data = response.content
    elif img_url.startswith("data:"):
        img_data = DataURI(img_url).data
    else:
        img_data = base64.b64decode(img_url)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def get_messages_pure(messages: list[ChatMessage]):
    images = []

    if messages and messages[-1].role == "assistant":
        messages.pop(-1)

    for m in messages:
        if m.role == "user":
            text = ""
            has_image = False

            for c in m.content:
                if c.type == "image_url":
                    images.extend([url_to_image(c.image_url.url)])
                    has_image = True
    return images
