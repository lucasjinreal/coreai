
import torch


class VLBase:
    def __init__(self, model_path=None, processor_path=None, device="auto"):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu" if device == "auto" else device
            )
        )
        self.model = self.load_model(model_path)
        self.processor = self.load_processor(
            processor_path if processor_path is None else model_path
        )

        self.history_msgs = []

    def load_model(self, model_path):
        raise NotImplementedError

    def load_processor(self, processor_path):
        raise NotImplementedError

    def stream_chat_with_images(self):
        raise NotImplementedError

    def generate(self, prompt, image, verbose):
        pass
