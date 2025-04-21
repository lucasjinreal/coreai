from coreai.models.firered.fireredasr import FireRedAsr
from audiotool.utils import normalize_audio_to_target


model = FireRedAsr.from_pretrained(model_dir="checkpoints/fireredasr-aed-l")

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["data/877.75_879.87.wav"]
a = normalize_audio_to_target(batch_wav_path[0])
res = model.transcribe(
    batch_uttid,
    [a],
    {
        "use_gpu": 0,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.0,
        "aed_length_penalty": 0.0,
        "eos_penalty": 1.0,
    },
)
print(res)
