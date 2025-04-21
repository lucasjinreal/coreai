from typing import Any, Dict, List, Literal, Optional
from dataclasses import asdict, dataclass, field
from transformers import HfArgumentParser

"""
default generating args for API demo.
"""


@dataclass
class GeneratingArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    """

    do_sample: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(
        default=0.3,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: Optional[int] = field(default=-1)
    num_beams: Optional[int] = field(default=1)
    max_new_tokens: Optional[int] = field(default=1200)
    repetition_penalty: Optional[float] = field(default=1.1)

    # length_penalty: Optional[float] = field(default=1.01)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="checkpoints/baichuan-7B")
    lora_model: Optional[str] = field(default="")
    model_name: Optional[str] = field(default="jarvis")
    conv_template: Optional[str] = field(default="qwen")
    int8: Optional[bool] = field(default=False)
    embedding: Optional[bool] = field(default=False)

    ip: Optional[str] = field(default=None)
    port: Optional[int] = field(default=8000)
    no_https: Optional[bool] = field(default=False)
    gpu_frac: Optional[float] = field(default=0.5)
    debug: Optional[bool] = field(default=False)
    do_embedding: Optional[bool] = field(default=False)
    negative_prompt: Optional[str] = field(default=None)
    guidance_scale: Optional[float] = field(default=1.0)


def get_generating_args():
    parser = HfArgumentParser((ModelArguments, GeneratingArguments))
    model_args, generating_args = parser.parse_args_into_dataclasses()
    return model_args, generating_args
