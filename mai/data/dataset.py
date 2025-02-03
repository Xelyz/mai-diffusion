import librosa
import torch
import sqlite3

import audioread.ffdec

import soundfile
import yaml
from pytorch_lightning import Callback
from torch.utils.data import Dataset
import cv2
import hashlib
from pytorch_lightning.utilities import rank_zero_only
import sys
import os
import csv
sys.path.append(os.getcwd())

from mai import util
from mai.data.convertor import *

class MaimaiDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_file,
                 feature_yaml=None,
                 sr=22050,
                 n_fft=2048,
                 max_audio_frame=16384,
                 audio_note_window_ratio=2,
                 n_mels=128,
                 mirror_p=0,
                 shift_p=0,
                 rate_p=0,
                 pitch_p=0,
                 feature_dropout_p=0,
                 freq_mask_p=0,
                 freq_mask_num=15,
                 rate=None,
                 test_csv_file=None,
                 with_audio=False,
                 with_feature=False,
                 cache_dir=None
                 ):
        # self.data_paths = txt_file
        # if isinstance(txt_file, str):
        #     txt_file_paths = [txt_file]
        # else:
        #     txt_file_paths = txt_file
        self.data_dir = data_dir

        with open(csv_file, 'r') as file:
            # csv contains json file name of chart, preprocessed by majdata edit, and audio file name
            # along with the features: style, diff, cc, name
            data = list(csv.DictReader(file))

        self.data = self.filter_beatmap_paths(data)

        self.feature_yaml = None
        self.with_feature = with_feature
        self.feature_dropout_p = feature_dropout_p
        if feature_yaml is not None and with_feature:
            self.feature_yaml = yaml.safe_load(open(feature_yaml))

        if test_csv_file is not None:
            with open(test_csv_file, "r", encoding='utf-8') as f:
                test_data = list(csv.DictReader(f))
                self.data = test_data + self.data

        self.audio_hop_length = n_fft // 4
        self.audio_frame_duration = self.audio_hop_length / sr
        self.audio_note_window_ratio = audio_note_window_ratio
        self.convertor_params = {
            "frame_ms": self.audio_frame_duration * audio_note_window_ratio * 1000,
            "max_frame": max_audio_frame // audio_note_window_ratio
        }
        self.mirror_p = mirror_p
        self.shift_p = shift_p
        self.rate_p = rate_p
        self.pitch_p = pitch_p
        self.freq_mask_p = freq_mask_p
        self.freq_mask_num = freq_mask_num
        self.with_audio = with_audio
        self.rate = rate
        self.sr = sr
        self.n_mels = n_mels
        self.max_audio_frame = max_audio_frame
        self.n_fft = n_fft
        self.max_duration = self.audio_frame_duration * max_audio_frame
        self.cache_dir = cache_dir
        self.error_files = []
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            # os.makedirs(os.path.join(cache_dir, 'processed'), exist_ok=True)
            error_path = os.path.join(self.cache_dir, "error.txt")
            if os.path.isfile(error_path):
                self.error_files = list(map(lambda x: x.strip(), open(error_path).readlines()))

    def __len__(self):
        return len(self.data)

    def load_feature(self, feature_dict, objs, dropout_prob=0, rate=1.0):
        if rate != 1.0:
            cc_ratio = 0.44 * (rate - 1) + 1
            feature_dict['cc'] *= cc_ratio

        slide_count = 0
        total = 0
        for timestamp, obj in objs.items():
            if obj.get('noteType') == 1:
                slide_count += 1
            total += 1
        
        slide_ratio = slide_count / total
        feature_dict['slide_ratio'] = slide_ratio
        if slide_ratio <= 0.08:
            feature_dict['keybr'] = 1
        elif 0.08 < slide_ratio <= 0.2:
            feature_dict['normal'] = 1
        else:
            feature_dict['stars'] = 1

        # dropout feature
        feature_dict_dropout = {}
        for k in feature_dict:
            if random.random() >= dropout_prob:
                feature_dict_dropout[k] = feature_dict[k]

        emb_ids = util.feature_dict_to_embedding_ids(feature_dict_dropout, self.feature_yaml)
        # print(f"{path} -> {emb_ids}")
        return feature_dict_dropout, emb_ids

    def __getitem__(self, i):
        song_data = self.data[i]
        path = os.path.join(self.data_dir, song_data['path'])
        chart_path = os.path.join(path, f"{song_data['diff']}.json")
        audio_path = os.path.join(path, "track.mp3")

        convertor_params = self.convertor_params.copy()
        convertor_params["mirror"] = np.random.random() < self.mirror_p
        convertor_params["offset_ms"] = 0
        convertor_params["rate"] = 1.0
        if self.rate is not None and np.random.random() < self.rate_p:
            # assert not self.with_audio, "Cannot change audio rate currently!"
            convertor_params["rate"] = np.random.random() * (self.rate[1] - self.rate[0]) + \
                                       self.rate[0]
        if np.random.random() < self.shift_p:
            assert not self.with_audio, "Cannot shift audio currently!"
            convertor_params["offset_ms"] = random.randint(0, int(convertor_params["max_frame"] *
                                                                  convertor_params["frame_ms"] / 2))
        try:
            objs, beatmap_meta = get_maimai_data(chart_path, audio_path, song_data, convertor_params)
            obj_array, valid_flag = beatmap_meta.convertor.objects_to_array(objs)
            example = {
                "meta": beatmap_meta.for_batch(),
                "convertor": convertor_params,
                "note": obj_array,
                "valid_flag": valid_flag
            }
            if self.with_audio:
                audio = util.load_audio(
                    self.cache_dir, beatmap_meta.audio, self.n_mels, self.audio_hop_length,
                    self.n_fft, self.sr, self.max_duration
                ).astype(np.float32)

                if convertor_params["rate"] != 1.0:
                    t = int(round(audio.shape[1] / convertor_params["rate"]))
                    audio = cv2.resize(audio.reshape(self.n_mels, -1, 1), (t, self.n_mels))
                
                t = audio.shape[1]
                if t < self.max_audio_frame:
                    audio = np.concatenate([
                        audio,
                        np.zeros((self.n_mels, self.max_audio_frame - t), dtype=np.float32)
                    ], axis=1)
                elif t > self.max_audio_frame:
                    audio = audio[:, :self.max_audio_frame]
                
                max_length_ms = np.sum(valid_flag) * convertor_params['frame_ms'] + 2000
                max_valid_length = int(max_length_ms / self.audio_frame_duration / 1000) + 1
                if max_valid_length < audio.shape[1]:
                    audio[:, max_valid_length:] = 0
                
                if np.random.random() < self.freq_mask_p:
                    f = int(np.random.uniform(0, self.freq_mask_num)) # [0, F)
                    f0 = random.randint(0, self.n_mels - f) # [0, v - f)
                    audio[f0:f0 + f, :] = 0
                
                if np.random.random() < self.pitch_p:
                    i = np.random.randint(1, 5)
                    zeros = np.zeros((i, audio.shape[1])).astype(np.float16)
                    if np.random.random() < 0.5:
                        audio = np.concatenate([audio[i:, :], zeros], axis=0)
                    else:
                        audio = np.concatenate([zeros, audio[:-i, :]], axis=0)

                example["audio"] = audio.astype(np.float32)

            if self.with_feature:
                feature_dict, feature = self.load_feature(beatmap_meta.features, objs, self.feature_dropout_p, convertor_params["rate"])
                example["feature"] = np.asarray(feature)
            return example
        except Exception as e:
            if path not in self.error_files:
                with open(os.path.join(self.cache_dir, "error.txt"), "a+") as f:
                    f.write(f"{path}: {e}\n")
                self.error_files.append(path)
            # raise
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths


class MaimaiTrainDataset(MaimaiDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths[:int(len(beatmap_paths) * 0.9)]


class MaimaiValidDataset(MaimaiDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_beatmap_paths(self, beatmap_paths):
        return beatmap_paths[int(len(beatmap_paths) * 0.9):]

# Needs Rewrite
class BeatmapLogger(Callback):
    def __init__(self, log_batch_idx, count, splits=None, log_images_kwargs=None):
        super().__init__()
        self.log_batch_idx = log_batch_idx
        self.splits = splits
        self.count = count
        if log_images_kwargs is None:
            log_images_kwargs = {}
        self.log_images_kwargs = log_images_kwargs

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if batch_idx not in self.log_batch_idx:
            return
        if split not in self.splits:
            return
        if not hasattr(pl_module, "log_beatmap") or not callable(pl_module.log_beatmap):
            return
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        pl_module.log_beatmap(batch, split=split, count=self.count, **self.log_images_kwargs)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwarg):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                *args, **kwarg):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    pass
    # import yaml
    # random.seed(0)
    # dataset = MaimaiDataset(txt_file="data/beatmap_4k/beatmap.txt", n_fft=512, max_audio_frame=32768, audio_note_window_ratio=8, 
    # n_mels=128, cache_dir="data/audio_cache", with_audio=True, with_feature=True, 
    # feature_yaml="configs/mug/mania_beatmap_features.yaml"
    # )
    # breakpoint()
    # dataset[0]


    
    # from tqdm import tqdm

    # os.makedirs(os.path.join("data", "audio_cache"), exist_ok=True)
    # base = (os.path.join("data", "audio_cache"))
    # for name in tqdm(os.listdir(base)):
    #     if name.endswith("npz"):
    #         y = np.load(os.path.join(base, name))['y']
    #         y = np.log1p(y).astype(np.float16)
    #         np.savez_compressed(os.path.join(os.path.join("data", "audio_cache_log_16"), name), y=y)
