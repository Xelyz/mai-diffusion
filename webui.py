import random

print("Loading imports")

import warnings
import subprocess
import torch
import minacalc
import threading
import requests

# print error if cuda fails to init
try:
    torch.cuda.init()
except:
    import traceback
    traceback.print_exc()

cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("WARNING: CUDA GPU is not available. Fallback to CPU mode, which may be slow!")

warnings.filterwarnings('ignore')

import zipfile
import base64
from collections import OrderedDict

import audioread.ffdec

import os

import eyed3
import gradio as gr
from omegaconf import OmegaConf
from reamber.algorithms.playField import PlayField
from reamber.algorithms.playField.parts import *
from reamber.osu.OsuMap import OsuMap

from mug.data.convertor import save_osu_file, parse_osu_file
from mug.data.utils import gridify, remove_intractable_mania_mini_jacks
from mug.diffusion.ddim import DDIMSampler
from mug.diffusion.diffusion import DDPM
from mug.util import feature_dict_to_embedding_ids, \
    load_audio_without_cache
from mug.diffusion.unet import *
from mug.firststage.autoencoder import *
from mug.cond.feature import *
from mug.cond.wave import *


def load_model_from_config(config, ckpt, verbose=False):
    global cuda_available
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if cuda_available:
        try:
            model.cuda()
        except:
            import traceback
            traceback.print_exc()
            cuda_available = False
            model.cpu()
            print(
                "WARNING: CUDA GPU is not available. Fallback to CPU mode, which may be slow!"
            )

    model.eval()
    return model


# TODO: make configurable
config = OmegaConf.load("models/ckpt/model.yaml")

try:
    result = requests.get(
        f"https://mugdiffusion.keytoix.vip/mugdiffusion/api/update?version={config.version}",
        timeout=(10, 10)
    ).json()

    if result['version'] > config.version:
        print(f"New version {result['version']} found! Please visit {result['incremental_url']} "
              f"for updating. \n"
              f"Press <Enter> to continue...")
        input()
except:
    pass


model = load_model_from_config(config, "models/ckpt/model.ckpt")
device = torch.device("cuda") if cuda_available else torch.device("cpu")
model = model.to(device)
template_path = "asset/template.osu"
output_path = "outputs/beatmaps/"
sampler = DDIMSampler(model, device)


def getHeight(y, ratio):
    left = 1
    right = y
    while (left <= right):
        if left == right:
            return left
        mid = int((left + right) / 2)
        res = (86 * np.ceil(y / mid) - 3) / mid
        if ratio > res:
            right = mid
        elif ratio < res:
            left = mid + 1
        else:
            return mid


ffmpeg_available = audioread.ffdec.available()
if not ffmpeg_available:
    print(
        "WARNING: ffmpeg not found. Please install ffmpeg first, otherwise the audio parsing may fail.")


def generate_feature_dict(audioPath, audioTitle, audioArtist,
                          rss, rs, srs, sr, etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc,
                          sss, ss, ssss, sssc, jss, js, jsss, jssc,
                          hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc, techs, tech, techss, techsc,
                          mts, lnrs, mapType, lnr, count, step, scale, rm_jacks, auto_snap, seed):
    # Very unreadable
    feature_dict = {}
    human_readable_dict = OrderedDict()

    def add_value_if(condition, key, val, h_key, h_val):
        if condition:
            feature_dict[key] = val
            human_readable_dict[h_key] = h_val

    add_value_if(rss, 'rank_status', 'ranked' if rs == 'ranked/stable' else rs, 'style', rs)
    add_value_if(srs, 'sr', sr, 'sr', sr)
    add_value_if(etts, 'ett', ett, 'msd', ett)

    if mts:
        if mapType.startswith("Rice"):
            feature_dict['rc'] = 1
            human_readable_dict["type"] = 'rc'
        elif mapType.startswith("Long Note"):
            feature_dict['ln'] = 1
            human_readable_dict["type"] = 'ln'
        elif mapType.startswith("Hybrid"):
            feature_dict['hb'] = 1
            human_readable_dict["type"] = 'hb'
        else:
            raise ValueError(mapType)

    add_value_if(lnrs, 'ln_ratio', lnr, "ln", lnr)

    patterns = [
        (cjs, 'Chordjack', cj, cjss, cjsc),
        (stas, 'Stamina', sta, stass, stasc),
        (sss, 'Stream', ss, ssss, sssc),
        (jss, 'Jumpstream', js, jsss, jssc),
        (hss, 'Handstream', hs, hsss, hssc),
        (jsps, 'Jackspeed', jsp, jspss, jspsc),
        (techs, 'Technical', tech, techss, techsc)
    ]
    for pattern_switch, pattern_name, pattern_value, pattern_score_switch, pattern_score_value in patterns:
        add_value_if(pattern_switch, pattern_name.lower(), pattern_value.startswith("more"),
                     pattern_name.lower(), "more" if pattern_value.startswith("more") else "less")
        # add_value_if(pattern_switch, pattern_name.lower(), pattern_name)
        add_value_if(pattern_score_switch, pattern_name.lower() + "_ett", pattern_score_value,
                     pattern_name.lower() + "-msd", pattern_score_value)

    human_readable_dict['rm-interval'] = rm_jacks
    human_readable_dict['snapping'] = auto_snap
    human_readable_dict['count'] = count
    human_readable_dict['step'] = step
    human_readable_dict['cfg-scale'] = scale
    human_readable_dict['seed'] = seed
    return feature_dict, human_readable_dict

def parse_feature(batch_size, feature_dicts, feature_yaml, model: DDPM):
    '''
    return model.model.cond_stage_model(feature)
    '''

    features = []
    for i in range(batch_size):
        # Why using cur_dict?
        cur_dict = feature_dicts
        features.append(feature_dict_to_embedding_ids(cur_dict, feature_yaml))
    feature = torch.tensor(np.asarray(features), dtype=torch.float32,
                           device=model.device)
    return model.model.cond_stage_model(feature)


def startInvertion(chart_path, rate):
    if rate is None or rate <= 0:
        raise gr.Error("Rate must > 0")
    if chart_path is None:
        raise gr.Error("Chart not found")

    hit_objects, meta = parse_osu_file(chart_path.name, None)
    notes = []
    ln = 0
    rc = 0

    for line in hit_objects:
        if line.strip() == "":
            continue
        try:
            params = line.split(",")
            if int(params[3]) == 128:
                ln += 1
            else:
                rc += 1
            start = int(float(params[2]))
            column = int(int(float(params[0])) / int(512 / 4))
            assert column <= 3
            notes.append((start, column))
        except:
            pass

    ln_ratio = ln / (ln + rc)

    is_rc = ln_ratio < 0.1
    is_ln = ln_ratio >= 0.4
    is_hb = 0.1 <= ln_ratio <= 0.4
    if is_rc:
        map_type = "Rice (LN < 10%)"
    elif is_hb:
        map_type = "Hybrid (10% < LN < 40%)"
    else:
        map_type = "Long Note (LN > 40%)"

    notes = sorted(notes, key=lambda x: x[0])

    # What is this?
    ett_result = minacalc.calc_skill_set(rate, notes)

    keys = [
        "overall",
        "stream",
        "jumpstream",
        "handstream",
        "stamina",
        "jackspeed",
        "chordjack",
        "technical",
    ]
    ett_result = dict(zip(keys, ett_result))
    result_patterns = ett_result.copy()
    del result_patterns['overall']
    del result_patterns['stamina']
    max_score = max(result_patterns.values())

    result = [
        ("MSD score (Etterna)", ett_result['overall']),
        ("map type", map_type),
        ("ln ratio", ln_ratio),
        ("chordjack", "more" if (max_score - ett_result['chordjack'] <= 1) else "less"),
        ("chordjack MSD", ett_result['chordjack']),
        ("stamina", "more" if (max_score - ett_result['stamina'] <= 1) else "less"),
        ("stamina MSD", ett_result['stamina']),
        ("stream", "more" if (max_score - ett_result['stream'] <= 1) else "less"),
        ("stream MSD", ett_result['stream']),
        ("jumpstream", "more" if (max_score - ett_result['jumpstream'] <= 1) else "less"),
        ("jumpstream MSD", ett_result['jumpstream']),
        ("handstream", "more" if (max_score - ett_result['handstream'] <= 1) else "less"),
        ("handstream MSD", ett_result['handstream']),
        ("jackspeed", "more" if (max_score - ett_result['jackspeed'] <= 1) else "less"),
        ("jackspeed MSD", ett_result['jackspeed']),
        ("technical", "more" if (max_score - ett_result['technical'] <= 1) else "less"),
        ("technical MSD", ett_result['technical']),
    ]
    # result = "\n".join(map(lambda x: f"{x[0]}: {x[1]}", result))
    return gr.update(value=result)


def startMapping(audioPath, audioTitle, audioArtist,
                 rss, rs, srs, sr, etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss,
                 ssss, sssc, jss, js, jsss, jssc,
                 hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc, techs, tech, techss, techsc, mts,
                 lnrs, mapType, lnr, count, step, scale, rm_jack_interval, auto_snap, seed,
                 progress=gr.Progress()):
    if cuda_available:
        torch.cuda.empty_cache()

    if audioPath is None:
        raise gr.Error("Audio not found!")

    audioPath = audioPath.name

    if not os.path.isfile(audioPath):
        raise gr.Error(f"Audio not found: {audioPath}")

    if audioTitle is None or audioTitle.strip() == "":
        raise gr.Error("Please specify your audio title")

    if audioArtist is None or audioArtist.strip() == "":
        raise gr.Error("Please specify your audio artist")

    # set seed
    seed = int(seed)
    if seed < 0:
        seed = random.randint(1, 100000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    try:
        with torch.no_grad():
            uc = None

            for progress_step in progress.tqdm(range(3), desc='Process prompts and audio'):
                if progress_step == 0:
                    feature_dict, h_dict = generate_feature_dict(
                        audioPath, audioTitle, audioArtist, rss, rs, srs, sr,
                        etts, ett, cjs, cj, cjss, cjsc, stas, sta, stass, stasc, sss, ss, ssss,
                        sssc,
                        jss, js, jsss, jssc, hss, hs, hsss, hssc, jsps, jsp, jspss, jspsc,
                        techs, tech, techss, techsc, mts, lnrs, mapType, lnr, count, step, scale,
                        rm_jack_interval,
                        auto_snap, seed
                    )
                    def prompt_mapping(item):
                        return f'{item[0]}={item[1]}'
                    prompt = ", ".join(map(prompt_mapping, h_dict.items()))
                    feature_yaml = yaml.safe_load(open("configs/mug/mania_beatmap_features.yaml"))

                    if scale != 1.0:
                        uc = parse_feature(count, {}, feature_yaml, model)

                    c = parse_feature(count, feature_dict, feature_yaml, model)

                elif progress_step == 1:
                    dataset = config.data.params.common_params

                    audio_hop_length = dataset.n_fft // 4
                    audio_frame_duration = audio_hop_length / dataset.sr

                    audio = load_audio_without_cache(audioPath, dataset.n_mels, dataset.n_fft // 4,
                                                     dataset.n_fft, dataset.sr,
                                                     None)

                elif progress_step == 2:
                    t = audio.shape[1]

                    audio_map_length_ratio = dataset.max_audio_frame // model.z_length  # 64
                    test_map_length = t / audio_map_length_ratio
                    test_map_length = (int(test_map_length / 32) + 1) * 32  # ensure the multiple
                    test_audio_length = test_map_length * audio_map_length_ratio
                    dataset.max_audio_frame = test_audio_length
                    model.z_length = test_map_length

                    # TODO: insert transformer mask
                    # padding or trunc audio to max_audio_frame
                    if t < dataset.max_audio_frame:
                        audio = np.concatenate([
                            audio,
                            np.zeros((dataset.n_mels, dataset.max_audio_frame - t),
                                     dtype=np.float32)
                        ], axis=1)
                    elif t > dataset.max_audio_frame:
                        audio = audio[:, :dataset.max_audio_frame]

                    w = torch.tensor(
                        np.stack([audio for _ in range(count)]),
                        dtype=torch.float32).to(model.device)
                    if cuda_available:
                        model.model.wave_model.to('cuda')
                    w = model.model.wave_model(w)
                    if cuda_available:
                        model.model.wave_model.to('cpu')
                        torch.cuda.empty_cache()

            shape = None
            samples_ddim, _ = sampler.sample(S=step,
                                             c=c, w=w,
                                             batch_size=count,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=0.0,
                                             tqdm_class=progress.tqdm)
            
            print(samples_ddim)
            # reamber generate example
            x_samples_ddim = model.model.decode(samples_ddim).cpu().numpy()
            print(x_samples_ddim)

            save_name = f"{audioArtist} - {audioTitle}"
            save_dir = os.path.join(output_path, save_name)
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir, exist_ok=True)
            convertor_params = {
                "frame_ms": audio_frame_duration * dataset.audio_note_window_ratio * 1000,
                "max_frame": dataset.max_audio_frame // dataset.audio_note_window_ratio
            }

            def custom_gridify(hit_objects):
                new_hit_objects, bpm, offset = gridify(hit_objects, verbose=False)
                if auto_snap:
                    hit_objects = new_hit_objects
                hit_objects = remove_intractable_mania_mini_jacks(hit_objects, verbose=False,
                                                                  jack_interval=rm_jack_interval)
                return bpm, offset, hit_objects

            previews = []

            convertor_params = convertor_params.copy()
            convertor_params["from_logits"] = True
            _, beatmap_meta = parse_osu_file(template_path, convertor_params)
            output_name = f"audio.mp3"

            proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                                     '-i', audioPath, '-c:a', 'libmp3lame',
                                     os.path.join(save_dir, output_name)
                                     ])
            proc.wait()
            if proc.returncode != 0:
                print("WARNING: cannot convert to mp3. Copy instead.")
                output_name = f"audio{os.path.splitext(audioPath)[-1]}"
                shutil.copyfile(audioPath, os.path.join(save_dir, output_name))

            for i, x_sample in enumerate(progress.tqdm(x_samples_ddim, desc='Post process charts')):
                version = f"AI v{i + 1}"
                creator = f"MuG Diffusion v{config.version}"
                file_name = f"{audioArtist} - {audioTitle} ({creator}) [{version}].osu".replace("/",
                                                                                                "")
                file_path = os.path.join(save_dir, file_name)

                save_osu_file(beatmap_meta, x_sample,
                              path=file_path,
                              override={
                                  "Creator": creator,
                                  "Version": version,
                                  "AudioFilename": output_name,
                                  "Title": audioTitle,
                                  "TitleUnicode": audioTitle,
                                  "Artist": audioArtist,
                                  "ArtistUnicode": audioArtist,
                                  "AIMode": creator,
                                  "AIPrompt": prompt + f", {i}"
                              }, gridify=custom_gridify)
                shutil.copyfile("asset/bg.jpg", os.path.join(save_dir, "bg.jpg"))

                # reamber generate example
                m = OsuMap.read_file(file_path)
                pf = (
                        PlayField(m=m, duration_per_px=5, padding=40) +
                        PFDrawBpm() +
                        PFDrawBeatLines() +
                        PFDrawColumnLines() +
                        PFDrawNotes() +
                        PFDrawOffsets()
                )
                originalHeight = pf.export().height
                processedHeight = getHeight(originalHeight, float(3.3))
                pic = pf.export_fold(max_height=processedHeight)
                previews.append(pic)
            # package
            output_osz_path = os.path.join(output_path, save_name + '.osz')
            with zipfile.ZipFile(output_osz_path, 'w') as f:
                for p in os.listdir(save_dir):
                    f.write(os.path.join(save_dir, p), arcname=p)

    except torch.cuda.OutOfMemoryError:
        raise gr.Error("Your GPU runs out of memory! "
                       "Please reopen MuG Diffusion and try to reduce the Sampling count, "
                       "or shrink the audio length. ")
    except (OSError, FileNotFoundError) as e:
        raise gr.Error(f"Your audio title or artist may contain strange characters that cannot "
                       f"serve as a file path. Reason: {e} ")

    return [
        gr.update(value=previews, visible=True), # char preview gallary
        gr.update(value=output_osz_path, visible=True),  # output file
        gr.update(value=f"{audioArtist} - {audioTitle}"),  # title
        gr.update(visible=True), # output file type
        gr.update(visible=True, value=prompt) # prompt overview
    ]


def generate_feature_dict_mai(audioPath, audioTitle, audioArtist, rs_switch, rs, diff_switch, diff, cc_switch, cc,
                    maptype_switch, slider_switch, mapType, slider, count, step, scale, auto_snap, seed):
    feature_dict = {}
    human_readable_dict = OrderedDict()

    def add_value_if(condition, key, val, h_key, h_val):
        if condition:
            feature_dict[key] = val
            human_readable_dict[h_key] = h_val

    add_value_if(rs_switch, 'style', rs, 'style', rs)
    add_value_if(diff_switch, 'diff', diff, 'diff', diff)
    add_value_if(cc_switch, 'cc', cc, 'cc', cc)

    if maptype_switch:
        if mapType.startswith("Keyboard"):
            feature_dict['keybr'] = 1
            human_readable_dict["type"] = 'keybr'
        elif mapType.startswith("Normal"):
            feature_dict['normal'] = 1
            human_readable_dict["type"] = 'normal'
        elif mapType.startswith("Stars"):
            feature_dict['stars'] = 1
            human_readable_dict["type"] = 'stars'
        else:
            raise ValueError(mapType)

    add_value_if(slider_switch, 'slide_ratio', slider, "slide", slider)

    # human_readable_dict['rm-interval'] = rm_jacks
    human_readable_dict['snapping'] = auto_snap
    human_readable_dict['count'] = count
    human_readable_dict['step'] = step
    human_readable_dict['cfg-scale'] = scale
    human_readable_dict['seed'] = seed
    return feature_dict, human_readable_dict

def startMapping_mai(audioPath, audioTitle, audioArtist, rs_switch, rs, diff_switch, diff, cc_switch, cc,
                maptype_switch, slider_switch, mapType, slider, count, step, scale, auto_snap,
                seed, progress=gr.Progress()):
    if cuda_available:
        torch.cuda.empty_cache()

    if audioPath is None:
        raise gr.Error("Audio not found!")

    audioPath = audioPath.name

    if not os.path.isfile(audioPath):
        raise gr.Error(f"Audio not found: {audioPath}")

    if audioTitle is None or audioTitle.strip() == "":
        raise gr.Error("Please specify your audio title")

    if audioArtist is None or audioArtist.strip() == "":
        raise gr.Error("Please specify your audio artist")

    # set seed
    seed = int(seed)
    if seed < 0:
        seed = random.randint(1, 100000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    try:
        with torch.no_grad():
            uc = None

            for progress_step in progress.tqdm(range(3), desc='Process prompts and audio'):
                if progress_step == 0:
                    feature_dict, h_dict = generate_feature_dict_mai(
                        audioPath, audioTitle, audioArtist, rs_switch, rs, diff_switch, diff, cc_switch, cc,
                        maptype_switch, slider_switch, mapType, slider, count, step, scale, auto_snap, seed
                    )
                    def prompt_mapping(item):
                        return f'{item[0]}={item[1]}'
                    prompt = ", ".join(map(prompt_mapping, h_dict.items()))
                    feature_yaml = yaml.safe_load(open("configs/mug/maimai_beatmap_features.yaml"))

                    if scale != 1.0:
                        uc = parse_feature(count, {}, feature_yaml, model)

                    c = parse_feature(count, feature_dict, feature_yaml, model)

                elif progress_step == 1:
                    dataset = config.data.params.common_params

                    audio_hop_length = dataset.n_fft // 4
                    audio_frame_duration = audio_hop_length / dataset.sr

                    audio = load_audio_without_cache(audioPath, dataset.n_mels, dataset.n_fft // 4,
                                                     dataset.n_fft, dataset.sr,
                                                     None)

                elif progress_step == 2:
                    t = audio.shape[1]

                    audio_map_length_ratio = dataset.max_audio_frame // model.z_length  # 64
                    test_map_length = t / audio_map_length_ratio
                    test_map_length = (int(test_map_length / 32) + 1) * 32  # ensure the multiple
                    test_audio_length = test_map_length * audio_map_length_ratio
                    dataset.max_audio_frame = test_audio_length
                    model.z_length = test_map_length

                    # TODO: insert transformer mask
                    # padding or trunc audio to max_audio_frame
                    if t < dataset.max_audio_frame:
                        audio = np.concatenate([
                            audio,
                            np.zeros((dataset.n_mels, dataset.max_audio_frame - t),
                                     dtype=np.float32)
                        ], axis=1)
                    elif t > dataset.max_audio_frame:
                        audio = audio[:, :dataset.max_audio_frame]

                    w = torch.tensor(
                        np.stack([audio for _ in range(count)]),
                        dtype=torch.float32).to(model.device)
                    if cuda_available:
                        model.model.wave_model.to('cuda')
                    w = model.model.wave_model(w)
                    if cuda_available:
                        model.model.wave_model.to('cpu')
                        torch.cuda.empty_cache()

            shape = None
            samples_ddim, _ = sampler.sample(S=step,
                                             c=c, w=w,
                                             batch_size=count,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=0.0,
                                             tqdm_class=progress.tqdm)
            # reamber generate example
            x_samples_ddim = model.model.decode(samples_ddim).cpu().numpy()

            save_name = f"{audioArtist} - {audioTitle}"
            save_dir = os.path.join(output_path, save_name)
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir, exist_ok=True)
            convertor_params = {
                "frame_ms": audio_frame_duration * dataset.audio_note_window_ratio * 1000,
                "max_frame": dataset.max_audio_frame // dataset.audio_note_window_ratio
            }

            def custom_gridify(hit_objects):
                new_hit_objects, bpm, offset = gridify(hit_objects, verbose=False)
                if auto_snap:
                    hit_objects = new_hit_objects
                # hit_objects = remove_intractable_mania_mini_jacks(hit_objects, verbose=False,
                #                                                   jack_interval=rm_jack_interval)
                return bpm, offset, hit_objects

            convertor_params = convertor_params.copy()
            convertor_params["from_logits"] = True
            _, beatmap_meta = parse_osu_file(template_path, convertor_params)
            output_name = f"audio.mp3"

            proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                                     '-i', audioPath, '-c:a', 'libmp3lame',
                                     os.path.join(save_dir, output_name)
                                     ])
            proc.wait()
            if proc.returncode != 0:
                print("WARNING: cannot convert to mp3. Copy instead.")
                output_name = f"audio{os.path.splitext(audioPath)[-1]}"
                shutil.copyfile(audioPath, os.path.join(save_dir, output_name))
            
            maidata = ''

            for i, x_sample in enumerate(progress.tqdm(x_samples_ddim, desc='Post process charts')):
                version = f"AI v{i + 1}"
                creator = f"MaiDiffusion v{config.version}"
                file_name = f"{audioArtist} - {audioTitle} ({creator}) [{version}].osu".replace("/",
                                                                                                "")
                file_path = os.path.join(save_dir, file_name)
                
                # create a save_maimai_file
                save_osu_file(beatmap_meta, x_sample,
                              path=file_path,
                              override={
                                  "Creator": creator,
                                  "Version": version,
                                  "AudioFilename": output_name,
                                  "Title": audioTitle,
                                  "TitleUnicode": audioTitle,
                                  "Artist": audioArtist,
                                  "ArtistUnicode": audioArtist,
                                  "AIMode": creator,
                                  "AIPrompt": prompt + f", {i}"
                              }, gridify=custom_gridify)
            shutil.copyfile("asset/bg.jpg", os.path.join(save_dir, "bg.jpg"))

            output_maidata_path = os.path.join(save_dir, 'maidata.txt')
            with open(output_maidata_path, 'w') as f:
                f.write(maidata)

            output_zip_path = os.path.join(output_path, save_name + '.zip')
            with zipfile.ZipFile(output_zip_path, 'w') as f:
                for p in os.listdir(save_dir):
                    f.write(os.path.join(save_dir, p), arcname=p)

    except torch.cuda.OutOfMemoryError:
        raise gr.Error("Your GPU runs out of memory! "
                       "Please reopen MuG Diffusion and try to reduce the Sampling count, "
                       "or shrink the audio length. ")
    except (OSError, FileNotFoundError) as e:
        raise gr.Error(f"Your audio title or artist may contain strange characters that cannot "
                       f"serve as a file path. Reason: {e} ")

    return [
        gr.update(value=output_zip_path, visible=True),  # output file
        gr.update(value=f"{audioArtist} - {audioTitle}"),  # title
        gr.update(visible=True), # output file type
        gr.update(visible=True, value=prompt) # prompt overview
    ]


if __name__ == "__main__":

    with gr.Blocks(title="MuG Diffusion") as webui:

        with open("asset/logo.png", "rb") as logo_file:
            encoded_string = base64.b64encode(logo_file.read()).decode('utf-8')

        gr.HTML(
            f'<div style="text-align: center; margin-bottom: 1rem">'
            f'<img src="data:image/png;base64,{encoded_string}" '
            f'style="width: 128px; height: 128px; margin: auto;"></img></div>'
            "<h1 style='text-align: center; margin-bottom: 1rem'>"
            "MuG Diffusion: High-quality and Controllable Charting AI for Rhythm Games"
            "</h1>"
            "<div style='text-align: center; margin-bottom: 1rem'> "
            "<a href='https://github.com/Keytoyze/Mug-Diffusion'>"
            "https://github.com/Keytoyze/Mug-Diffusion</a></div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                # audioPath = gr.Audio(label="Audio file", info="drop audio here", type="filepath")
                audioPath = gr.File(label="Audio file", type="filepath", file_types=['audio'])

            with gr.Column(scale=1):
                audioTitle = gr.Textbox(label="Audio title", lines=1)
                audioArtist = gr.Textbox(label="Audio artist", lines=1)


            def on_change_audio(x):
                try:
                    path = x['name']
                    audio_file = eyed3.load(path)
                    audio_artist = audio_file.tag.artist
                    audio_title = audio_file.tag.title
                except:
                    try:
                        audio_title = os.path.basename(x['name']).split('.')[0]
                    except:
                        audio_title = ""
                    audio_artist = ""
                return [
                    gr.update(value=audio_title),
                    gr.update(value=audio_artist),
                ]


            audioPath.change(on_change_audio, inputs=audioPath, outputs=[audioTitle, audioArtist],
                             preprocess=False, postprocess=False)

        with gr.Tab("Vertical Scroll Rhythm Game (4K)"):
            with gr.Row():

                with gr.Column(scale=1):
                    with gr.Accordion("Diffculty", open=True):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                sr_switch = gr.Checkbox(label="star rating (osu!)", value=True)
                                ett_switch = gr.Checkbox(label="MSD score (Etterna)")
                            with gr.Column(scale=3, min_width=100):
                                sr = gr.Slider(1, 8, value=4, label="star rating (osu!)")
                                ett = gr.Slider(5, 35, value=20, label="MSD (Etterna)", visible=False)


                        def etts_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def srs_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        ett_switch.select(etts_switch, None, ett)
                        sr_switch.select(srs_switch, None, sr)

                    with gr.Accordion("Rice & long note", open=True):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                maptype_switch = gr.Checkbox(label="map type")
                                lnr_switch = gr.Checkbox(label="ln ratio")
                            with gr.Column(scale=3, min_width=100):
                                mapType = gr.Radio(["Rice (LN < 10%)", "Long Note (LN > 40%)",
                                                    "Hybrid (10% < LN < 40%)"],
                                                   show_label=False, value="Rice (LN < 10%)",
                                                   visible=False)
                                lnr = gr.Slider(0, 1, value=0.0, label="ln ratio", visible=False,
                                                info="ln ratio of the map, 0 for rice only, 1 for FULL LN")


                        def mts_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def lnrs_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        maptype_switch.select(mts_switch, None, mapType)
                        lnr_switch.select(lnrs_switch, None, lnr)

                    with gr.Accordion("Special", open=True):
                        rm_jacks = gr.Number(label="minimal interval (ms) for eliminating minijacks",
                                             info="will rearrange patterns to reduce intractable "
                                                  "minijack less than the given interval. "
                                                  "By default, 90 ms is the 1/4 meter of "
                                                  "166.67 bpm (= 15,000 / 90).",
                                             value=90)
                        # rm_jacks = gr.Checkbox(label="remove intractable mini jacks",
                        #                        info="recommend when generating stream patterns",
                        #                        value=True)
                        auto_snap = gr.Checkbox(label="snap notes to grids",
                                                info="recommend when there are no bpm changes",
                                                value=True)

                    with gr.Accordion("Model configurations", open=True):
                        with gr.Row():
                            count = gr.Slider(1, 16, value=4.0, label="Sampling count", info="number of charts",
                                              step=1.0)
                            step = gr.Slider(10, 200, value=100, label="Sampling step",
                                             info="step of diffusion process", step=1.0)
                        with gr.Row():
                            scale = gr.Slider(1, 30, value=5.0, label="CFG scale",
                                              info="how strongly it should conform to prompt")
                            seed = gr.Number(value=-1, label='Random seed',
                                             info="random seed for reproduction. "
                                                  "-1 will cause a new seed.")



                with gr.Column(scale=1):
                    with gr.Accordion("Pattern", open=True):
                        gr.Markdown("NOTE: If you don't choose an option, AI will "
                                    "**freely decide the pattern**, "
                                    "instead of **not writing this pattern**. "
                                    "For precisely controlling, "
                                    'try to select the option and choose `more` or `less`.')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                rs_switch = gr.Checkbox(label="style", value=True, elem_id="lbox")
                            with gr.Column(scale=3, min_width=100):
                                rs = gr.Radio(['ranked/stable', 'loved', 'graveyard'], \
                                              value='ranked/stable', show_label=False)


                        def rss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        rs_switch.select(rss_switch, None, rs)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                cj_switch = gr.Checkbox(label="chordjack")
                                cj_score_switch = gr.Checkbox(label="chordjack MSD")
                            with gr.Column(scale=3, min_width=100):
                                cj = gr.Radio(['more chordjack', 'less chordjack'],
                                              value='more chordjack',
                                              visible=False, show_label=False)
                                cj_score = gr.Slider(5, 35, value=17, label="chordjack MSD:",
                                                     visible=False)


                        def cje_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def cjss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        cj_switch.select(cje_switch, None, cj)
                        cj_score_switch.select(cjss_switch, None, cj_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                sta_switch = gr.Checkbox(label="stamina")
                                sta_score_switch = gr.Checkbox(label="stamina MSD")
                            with gr.Column(scale=3, min_width=100):
                                sta = gr.Radio(['more stamina', 'less stamina'],
                                               value='more stamina', \
                                               visible=False, show_label=False)
                                sta_score = gr.Slider(5, 35, value=17, label="stamina MSD:",
                                                      visible=False)


                        def stae_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def stass_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        sta_switch.select(stae_switch, None, sta)
                        sta_score_switch.select(stass_switch, None, sta_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                ss_switch = gr.Checkbox(label="stream")
                                ss_score_switch = gr.Checkbox(label="stream MSD")
                            with gr.Column(scale=3, min_width=100):
                                ss = gr.Radio(['more stream', 'less stream'],
                                              value='more stream',
                                              visible=False, show_label=False)
                                ss_score = gr.Slider(5, 35, value=17, label="stream MSD:",
                                                     visible=False)


                        def sse_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def ssss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        ss_switch.select(sse_switch, None, ss)
                        ss_score_switch.select(ssss_switch, None, ss_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                js_switch = gr.Checkbox(label="jumpstream")
                                js_score_switch = gr.Checkbox(label="jumpstream MSD")
                            with gr.Column(scale=3, min_width=100):
                                js = gr.Radio(['more jumpstream', 'less jumpstream'],
                                              value='more jumpstream',
                                              visible=False, show_label=False)
                                js_score = gr.Slider(5, 35, value=17, label="jumpstream MSD:",
                                                     visible=False)


                        def jse_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def jsss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        js_switch.select(jse_switch, None, js)
                        js_score_switch.select(jsss_switch, None, js_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                hs_switch = gr.Checkbox(label="handsteam")
                                hs_score_switch = gr.Checkbox(label="handstream MSD")
                            with gr.Column(scale=3, min_width=100):
                                hs = gr.Radio(['more handstream', 'less handstream'],
                                              value='more handstream',
                                              visible=False, show_label=False)
                                hs_score = gr.Slider(5, 35, value=17, label="handsteam MSD:",
                                                     visible=False)


                        def hse_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def hsss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        hs_switch.select(hse_switch, None, hs)
                        hs_score_switch.select(hsss_switch, None, hs_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                jsp_switch = gr.Checkbox(label="jackspeed")
                                jsp_score_switch = gr.Checkbox(label="jackspeed MSD")
                            with gr.Column(scale=3, min_width=100):
                                jsp = gr.Radio(['more jackspeed', 'less jackspeed'],
                                               value='more jackspeed',
                                               visible=False, show_label=False)
                                jsp_score = gr.Slider(5, 35, value=17, label="jackspeed MSD:",
                                                      visible=False)


                        def jspe_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def jspss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        jsp_switch.select(jspe_switch, None, jsp)
                        jsp_score_switch.select(jspss_switch, None, jsp_score)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                tech_switch = gr.Checkbox(label="technical")
                                tech_score_switch = gr.Checkbox(label="technical MSD")
                            with gr.Column(scale=3, min_width=100):
                                tech = gr.Radio(['more technical', 'less technical'],
                                                value='more technical',
                                                visible=False, show_label=False)
                                tech_score = gr.Slider(5, 35, value=17, label="technical MSD:",
                                                       visible=False)


                        def teche_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def techss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        tech_switch.select(teche_switch, None, tech)
                        tech_score_switch.select(techss_switch, None, tech_score)

            btn = gr.Button('Start Generation', variant='primary')

            inp = [audioPath, audioTitle, audioArtist, rs_switch, rs, sr_switch, sr, ett_switch, ett,
                   cj_switch, cj, cj_score_switch, cj_score, sta_switch, sta, \
                   sta_score_switch, sta_score, ss_switch, ss, ss_score_switch, ss_score, js_switch, js,
                   js_score_switch, js_score, hs_switch, hs, \
                   hs_score_switch, hs_score, jsp_switch, jsp, jsp_score_switch, jsp_score, tech_switch,
                   tech, tech_score_switch, tech_score, \
                   maptype_switch, lnr_switch, mapType, lnr, count, step, scale, rm_jacks, auto_snap,
                   seed
                   ]

            prompt_out = gr.Markdown(visible=False)
            out_preview = gr.Gallery(label="Chart overview", visible=True, elem_id='output', preview=True, object_fit='fill')

            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    fileFormat = gr.Radio(['.osz'], show_label=False, value='.osz', visible=False)
                with gr.Column(scale=2, min_width=100):
                    out_file = gr.File(label='Output file', visible=False, interactive=False, elem_id='outputFile')

            def getFileName(audioArtist, audioTitle):
                return gr.update(value=f"{audioArtist} - {audioTitle}", visible=True)
            fileName = gr.Textbox(visible=False)

            fileFormat.change(None, [fileFormat, fileName], None, js='''(format, name) => {
                               var file = document.getElementById('outputFile');
                               file.querySelector('td').innerText = name+format;
                               file.querySelector('a').setAttribute('download', name+format);}
                               ''')

            btn.click(lambda: gr.update(visible=False), None, out_file)
            btn.click(lambda: gr.update(visible=False), None, fileFormat)
            btn.click(lambda: gr.update(visible=False), None, prompt_out)
            btn.click(startMapping, inp, [out_preview, out_file, fileName, fileFormat, prompt_out],
                      api_name='generate')

        with gr.Tab("Chart to Prompt (4K)"):
            # with gr.Column(scale=1):
            gr.Markdown("This tool can get prompts from a chart / beatmap file, which helps you "
                        "write better prompts for AI charting.")
            with gr.Row():
                with gr.Column(scale=1):
                    chart_path = gr.File(label="Chart file", type="filepath",
                                         file_types=['.osu'])
                    rate = gr.Number(value=1.0, label='Rate', precision=2)
                    invert_btn = gr.Button('Generate prompt', variant='primary')
                with gr.Column(scale=1):
                    invert_out = gr.Dataframe(headers=["Parameter", "Value"], interactive=False)
            invert_btn.click(startInvertion, [chart_path, rate], invert_out, api_name='chart2prompt')


        with gr.Tab("Maimai DX"):
            gr.Markdown("Working project: Generate maimai chart from audio")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Diffculty", open=True):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                diff_switch = gr.Checkbox(label="Difficulty", value=True)
                                cc_switch = gr.Checkbox(label="Chart Constant", value=True)
                            with gr.Column(scale=3, min_width=100):
                                diff = gr.Radio(["Basic", "Advanced", "Expert", "Master", "Re:MASTER"], value="Master")
                                cc = gr.Slider(1, 15, value=13, step=0.1, label="Chart Constant")


                        def diffs_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def ccs_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        diff_switch.select(diffs_switch, None, diff)
                        cc_switch.select(ccs_switch, None, cc)

                    with gr.Accordion("Slides", open=True):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                maptype_switch = gr.Checkbox(label="map type")
                                slider_switch = gr.Checkbox(label="slide ratio")
                            with gr.Column(scale=3, min_width=100):
                                mapType = gr.Radio(["Keyboard (Slide < 10%)", "Stars (Slide > 25%)",
                                                    "Normal (10% < Slide < 25%)"],
                                                    show_label=False, value="Normal (10% < Slide < 25%)",
                                                    visible=False)
                                slider = gr.Slider(0, 0.5, value=0.0, label="slide ratio", visible=False,
                                                info="slide ratio of the map, 0 for notes only, 0.5 for FULL slide"
                                                "(because slide always has a tap as head)")

                        def mts_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)


                        def sliders_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)

                        maptype_switch.select(mts_switch, None, mapType)
                        slider_switch.select(sliders_switch, None, slider)


                    with gr.Accordion("Special", open=True):
                        # rm_jacks = gr.Number(label="minimal interval (ms) for eliminating minijacks",
                        #                      info="will rearrange patterns to reduce intractable "
                        #                           "minijack less than the given interval. "
                        #                           "By default, 90 ms is the 1/4 meter of "
                        #                           "166.67 bpm (= 15,000 / 90).",
                        #                      value=90)
                        auto_snap = gr.Checkbox(label="snap notes to grids",
                                                        info="recommend when there are no bpm changes",
                                                        value=True)

                    with gr.Accordion("Model configurations", open=True):
                        with gr.Row():
                            count = gr.Slider(1, 16, value=4.0, label="Sampling count", info="number of charts",
                                                    step=1.0)
                            step = gr.Slider(10, 200, value=100, label="Sampling step",
                                                    info="step of diffusion process", step=1.0)
                        with gr.Row():
                            scale = gr.Slider(1, 30, value=5.0, label="CFG scale",
                                                    info="how strongly it should conform to prompt")
                            seed = gr.Number(value=-1, label='Random seed',
                                                    info="random seed for reproduction. "
                                                        "-1 will cause a new seed.")

                with gr.Column(scale=1):
                    with gr.Accordion("Pattern", open=True):
                        gr.Markdown("NOTE: If you don't choose an option, AI will "
                                            "**freely decide the pattern**, "
                                            "instead of **not writing this pattern**. "
                                            "For precisely controlling, "
                                            'try to select the option and choose `more` or `less`.')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=100):
                                rs_switch = gr.Checkbox(label="style", value=True, elem_id="lbox")
                            with gr.Column(scale=3, min_width=100):
                                rs = gr.Radio(['Standard', 'DX', 'Festival'], \
                                                    value='Festival', show_label=False)

                        def rss_switch(evt: gr.SelectData):
                            return gr.update(visible=evt.selected)

                        rs_switch.select(rss_switch, None, rs)



            btn = gr.Button('Start Generation', variant='primary')

            inp = [audioPath, audioTitle, audioArtist, rs_switch, rs, diff_switch, diff, cc_switch, cc,
                        maptype_switch, slider_switch, mapType, slider, count, step, scale, auto_snap,
                        seed
                        ]

            prompt_out = gr.Markdown(visible=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    fileFormat = gr.Radio(['.osz'], show_label=False, value='.osz', visible=False)
                with gr.Column(scale=2, min_width=100):
                    out_file = gr.File(label='Output file', visible=False, interactive=False, elem_id='outputFile')

            def getFileName(audioArtist, audioTitle):
                return gr.update(value=f"{audioArtist} - {audioTitle}", visible=True)
            fileName = gr.Textbox(visible=False)

            fileFormat.change(None, [fileFormat, fileName], None, js='''(format, name) => {
                            var file = document.getElementById('outputFile');
                            file.querySelector('td').innerText = name+format;
                            file.querySelector('a').setAttribute('download', name+format);}
                            ''')

            btn.click(lambda: gr.update(visible=False), None, out_file)
            btn.click(lambda: gr.update(visible=False), None, fileFormat)
            btn.click(lambda: gr.update(visible=False), None, prompt_out)
            btn.click(startMapping, inp, [out_file, fileName, fileFormat, prompt_out],
                    api_name='generate_mai')
            

        with gr.Tab("Other Modes (to be continue)"):
            pass

        gr.Markdown("Copyright: Charts created through MuG Diffusion are "
                    "fully open source, explicitly falling under "
                    "the [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) "
                    "Universal Public Domain Dedication.")

    webui.queue(10).launch(share=False, favicon_path='asset/logo.ico', inbrowser=True)

