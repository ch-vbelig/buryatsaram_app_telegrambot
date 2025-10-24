from models import Text2Mel, MELSSRN, HiFiGenerator
from config import Config, AttrDict
import text_processing as txt_processing
import torch
import torchaudio
import json
import io
import glob

# Get device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# Paths to states
text2mel_state_path = sorted(glob.glob('model_states/text2mel-checkpoint*.pth'))[-1]
melssrn_state_path = sorted(glob.glob('model_states/melssrn-checkpoint*.pth'))[-1]
vocoder_path = sorted(glob.glob('model_states/g_*'))[-1]
vocoder_config_path = './config_v1.json'


with open(vocoder_config_path) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)


# Restore config
state_t2m = torch.load(text2mel_state_path, map_location=device)
config_t2m = state_t2m["config"]

state_mel_ssrn = torch.load(melssrn_state_path, map_location=device)
config_ssrn = state_mel_ssrn["config"]

state_vocoder = torch.load(vocoder_path, map_location=device)

# Load networks
print("Loading Text2Mel...")
text2mel = Text2Mel().to(device)
text2mel.eval()
text2mel_step = state_t2m["global_step"]
text2mel.load_state_dict(state_t2m["model"])

print("Loading MELSSRN...")
mel_ssrn = MELSSRN().to(device)
mel_ssrn.eval()
mel_ssrn_step = state_mel_ssrn["global_step"]
mel_ssrn.load_state_dict(state_mel_ssrn["model"])

print("Loading Vocoder...")
vocoder = HiFiGenerator(h).to(device)
vocoder.load_state_dict(state_vocoder["generator"])
vocoder.eval()
vocoder.remove_weight_norm()


def synthesize(msg):
    text_ids, texts = txt_processing.normalize_long_text(msg, chunk_size=4)

    # Get max text and mel lengths in batch
    max_text_len = max(len(text) for text in text_ids)

    long_text_padded = torch.zeros(len(text_ids), max_text_len, dtype=torch.long)

    # Pad data
    for i in range(len(text_ids)):
        long_text_padded[i, :len(text_ids[i])] = torch.tensor(text_ids[i])

    L = torch.tensor(long_text_padded.clone().detach(), device=device, requires_grad=False)
    S_mel = torch.zeros(len(text_ids), Config.max_T_len, Config.F, requires_grad=False,
                        device=device)  # S: (bs, T, n_mels)
    previous_position = torch.zeros(len(text_ids), requires_grad=False, dtype=torch.long,
                                    device=device)  # tensor([0]) # (1)
    previous_att = torch.zeros(len(text_ids), max_text_len, Config.max_T_len, requires_grad=False,
                               device=device)  # 1, N, max_T

    flag = False
    with torch.no_grad():
        for t in range(Config.max_T_len - 1):
            # Y: (bs, n_mels, ts)
            _, Y, A, current_position = text2mel.forward(L, S_mel.transpose(1, 2),
                                                         force_incremental_att=True,
                                                         previous_att_position=previous_position,
                                                         previous_att=previous_att,
                                                         current_time=t,
                                                         flag=flag
                                                         )
            flag = not flag
            # S: (bs, T, n_mels)
            # Y (transposed): (bs, T, n_mels)
            Y = Y.transpose(1, 2).detach()[:, t, :]
            S_mel[:, t + 1, :] = Y
            previous_position = current_position.detach()
            previous_att = A.detach()

        # S_mel: (bs, max_T, 80) : (1, max_T, 80)
        # S_mel_full: (1, 80, max_T * 4)
        _, S_mel_full = mel_ssrn(S_mel.transpose(1, 2))

        # De-noramlize
        S_mel_full_d = (torch.clamp(S_mel_full, 0, 1) * 100) - 100 + 20

        # to amplitude
        S_mel_full_d = torch.pow(10.0, S_mel_full_d * 0.05)

        # wav: (bs, 1, n_samples)
        wav = vocoder(S_mel_full_d)

    wav_joined = torch.zeros(1, Config.sample_rate)
    trigger_level = 8
    for i in range(wav.size(0)):
        w = wav[i].detach().cpu()
        w = torchaudio.functional.vad(w.flip([-1]), Config.sample_rate, trigger_level=trigger_level).flip([-1])
        if texts[i][-2] in Config.signs[5:]:
            print(texts[i], 'adding pause')
            wav_joined = torch.cat((wav_joined, w, torch.zeros(1, Config.sample_rate // 3)), dim=1)
        else:
            wav_joined = torch.cat((wav_joined, w), dim=1)
            print(texts[i], 'not adding pause')


    wav_joined = torch.cat((wav_joined, torch.zeros(1, Config.sample_rate)), dim=1)
    wav = wav_joined.detach().cpu()

    # Convert the tensor to a WAV format using torchaudio
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, sample_rate=Config.sample_rate, format='wav')

    # Seek to the beginning of the stream
    buffer.seek(0)

    return buffer