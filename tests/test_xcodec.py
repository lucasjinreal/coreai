from coreai.tasks.audio.codecs.xcodec2.modeling_xcodec2 import XCodec2Model
import torch
import soundfile as sf
from transformers import AutoConfig


import torchaudio
import torch


def load_audio_mono_torchaudio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to numpy array
    wav = waveform.numpy().squeeze()
    return wav, sample_rate


model_path = "checkpoints/XCodec2_bf16"

model = XCodec2Model.from_pretrained(model_path)
model.eval()
# model.to(torch.bfloat16)
# model.save_pretrained("checkpoints/XCodec2_bf16")

# wav, sr = load_audio_mono_torchaudio("data/79.3_82.0.wav")
wav, sr = load_audio_mono_torchaudio("data/877.75_879.87.wav")
# wav, sr = sf.read("data/test.flac")
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)


with torch.no_grad():
    # vq_code = model.encode_code(input_waveform=wav_tensor)
    # print("Code:", vq_code)

    vq_code_fake = torch.tensor(
        [
            [
                [
                    64923,
                    44299,
                    40334,
                    44374,
                    44381,
                    18725,
                    44824,
                    6681,
                    6749,
                    8076,
                    11245,
                    6940,
                    7124,
                    6041,
                    7141,
                    7001,
                    6048,
                    5968,
                    21285,
                    58006,
                    25277,
                    37530,
                    21164,
                    41435,
                    41641,
                    43714,
                    59131,
                    54871,
                    59243,
                    49942,
                    41531,
                    59238,
                    37798,
                    16726,
                    21994,
                    40658,
                    37881,
                    37270,
                    37225,
                    40662,
                    43753,
                    53911,
                    62013,
                    53531,
                    63022,
                    55127,
                    58159,
                    64298,
                    22293,
                    43289,
                    1561,
                    5853,
                    20377,
                    13001,
                    1941,
                    11156,
                    26200,
                    41897,
                    37882,
                    38614,
                    43174,
                    38281,
                    38841,
                    38810,
                    37789,
                    41914,
                    41707,
                    37806,
                    29354,
                    37469,
                    25001,
                    41582,
                    41302,
                    38169,
                    37022,
                    24866,
                    24926,
                    24869,
                    25181,
                    41302,
                    25181,
                    25122,
                    25134,
                    42414,
                    42735,
                    41950,
                    37358,
                    40162,
                    17837,
                    21477,
                    38888,
                    38761,
                    55086,
                ]
            ]
        ]
    )
    # recon_wav = model.decode_code(vq_code).cpu()  # Shape: (1, 1, T')
    recon_wav = model.decode_code(vq_code_fake).cpu()  # Shape: (1, 1, T')


sf.write("data/reconstructed2.wav", recon_wav[0, 0, :].numpy(), sr)
print("Done! Check reconstructed.wav")
