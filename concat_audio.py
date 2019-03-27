from pydub import AudioSegment
import glob
import os

from data_loader import speakers
from preprocess import get_spk_world_feats


def concat_wavs(wavs, out_filepath):
    out = AudioSegment.from_wav(wavs[0])
    for wav in wavs[1:]:
        out += AudioSegment.from_wav(wav)
    out.export(out_filepath, format='wav')


def concat_for_all_speakers():
    for spk in speakers:
        path = 'data/VCTK-corpus/wav16/' + spk
        wavs = glob.glob(path + '/*.wav')
        out_dir = 'data/concatted_audio/wav/' + spk
        os.makedirs(out_dir, exist_ok=True)
        concat_wavs(wavs, out_dir + spk + '/' + spk + '_concatted.wav')


def create_mc():
    for spk in speakers:
        spk_fold_path = 'data/concatted_audio/wav/' + spk
        mc_dir_train = 'data/concatted_audio/mc/train'
        mc_dir_test = 'data/concatted_audio/mc/test'
        get_spk_world_feats(spk_fold_path, mc_dir_train, mc_dir_test, sample_rate=16000)


concat_for_all_speakers()
#create_mc()


