import inspect
import itertools
import os
import shutil

max_msg_len = 800

class Config:

    # P - symbol for padding
    # E - symbol for end of sentence

    vocab = 'PE абвгдеёжзийклмнопрстуфхцчшщьыъэюяһүө.,!?'
    vocab_bur_only = ' абвгдеёжзийклмнопрстуфхцчшщьыъэюяһүө.,!?'
    sounds_only = ' абвгдеёжзийклмнопрстуфхцчшщьыъэюяһүө'

    vowels = 'аеёиоуыэюяүө'
    consonants = 'бвгджзйклмнпрстфхцчшщһ'
    # signs = 'PE ьъ«»:–—!?.,-'
    signs = 'PE ьъ:!?.,-'

    vocab_padding_index = 0 # index of padding character
    vocab_end_of_text = "E"

    # Network dims
    e = 128 # embedding dim
    d = 256 # "d" for hidden unit of Text2Mel
    c = 512 # hidden units of SSRN

    F = 80 # n_mels
    F_ = 1025 # STFT spec frequency resolution (half of Nyquist F)

    y_factor = 0.6
    n_factor = 1.3

    num_bit = 10

    # Params for audio preprocessing
    sample_rate = 22050
    num_fft_samples = (F_-1) * 2 # =2048 resolution for spectrogram

    frame_length = 0.05 # seconds (50ms), original paper uses 0.012
    frame_shift = frame_length / 4 # seconds (12.5ms)

    # window_length = int(sample_rate * frame_length) # 1102
    # hop_length = int(sample_rate * frame_shift) # 276
    window_length = 1024
    hop_length = 256

    mel_size = F # n_mels
    power = 2.0
    preemphasis = 0.97

    n_iter = 50 # number of inversion iterations
    max_db = 100
    ref_db = 20

    time_reduction = 4 # only consider every n-th timestep

    # Params for training
    # Maximum number of characters & mel_frames
    # Ignore training samples that are too long
    # Ca be tweaked to limit memory consumption default 180, 210
    max_N, max_T = 180, 240

    # For synthesis
    max_N_len, max_T_len = 40, 60

    g = 0.2 # guided attention
    dropout_rate = 0.05


    @staticmethod
    def get_config():
        attributes = inspect.getmembers(Config, lambda  a: not(inspect.isroutine(a)))
        return  [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

    @staticmethod
    def set_config(a):
        for name, value in a:
            setattr(Config, name, value)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

