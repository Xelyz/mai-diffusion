import os
import sys
import random
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import string
import json
sys.path.append(os.getcwd())

from mai.data.utils import Touch, get_slide_path, HitObject, gridify, Beats

import numpy as np


@dataclass
class BeatmapMeta:
    name: str = ""
    chart: str = ""
    audio: str = ""
    features: dict = field(default_factory=lambda: {})
    convertor: 'BaseMaimaiConvertor' = None
    timing_points: List[str] = field(default_factory=lambda: [])

    def for_batch(self):
        result = asdict(self, dict_factory=lambda x: {k: v
                                                      for (k, v) in x
                                                      if k != 'convertor' and k != 'file_meta' and k != 'timing_points'})
        return result


def read_item(line):
    return line.split(":")[-1].strip()

valid_chars = "-_.()[]/\\' %s%s" % (string.ascii_letters, string.digits)

def slugify(text):
    return "".join(c for c in text if c in valid_chars)

def get_maimai_data(chart_path, audio_path, song_data, convertor_params: Optional[dict]) -> Tuple[dict, BeatmapMeta]:
    with open(chart_path, 'r') as file:
        data = json.load(file)

    meta = BeatmapMeta()
    meta.chart = chart_path
    meta.audio = audio_path
    meta.name = song_data['name']
    features_keys = ['style', 'diff', 'cc']
    meta.features = {key: song_data[key] for key in features_keys if key in song_data}

    if convertor_params is not None:
        meta.convertor = MaimaiConvertor(**convertor_params)

    return data, meta
    # for line in data:
    #     line = line.strip()

    #     if parsing_context == "[HitObjects]" and "," in line:
    #         hit_objects.append(line)
    #     elif parsing_context == "[TimingPoints]" and "," in line:
    #         meta.file_meta.append(line)
    #         meta.timing_points.append(line)
    #     else:
    #         if line != "[HitObjects]":
    #             meta.file_meta.append(line)

    #         if parsing_context == "[General]":
    #             if line.startswith("AudioFilename"):
    #                 audio_item = read_item(line)
    #                 meta.audio = os.path.join(os.path.dirname(osu_path),
    #                                           audio_item)
    #                 if not os.path.isfile(meta.audio):
    #                     meta.audio = os.path.join(os.path.dirname(osu_path),
    #                                               slugify(audio_item))
    #                     if not os.path.isfile(meta.audio):
    #                         meta.audio = os.path.join(
    #                             os.path.dirname(meta.audio),
    #                             audio_item.lower()
    #                         )
    #                         if not os.path.isfile(meta.audio):
    #                             meta.audio = os.path.join(
    #                                 os.path.dirname(meta.audio),
    #                                 slugify(audio_item.lower())
    #                             )

    #         elif parsing_context == "[Metadata]":
    #             if line.startswith("Version"):
    #                 meta.version = read_item(line)
    #             elif line.startswith("BeatmapSetID"):
    #                 meta.set_id = int(read_item(line))

    #         elif parsing_context == "[Difficulty]":
    #             if line.startswith("CircleSize"):
    #                 meta.cs = float(read_item(line))

    #     if line.startswith("["):
    #         parsing_context = line

def save_maimai_file(meta: BeatmapMeta, note_array: np.ndarray, path=None,
                  gridify=None):
    convertor = meta.convertor
    hit_objects = convertor.array_to_objects(note_array)
    try:
        bpm, offset, hit_objects = gridify(hit_objects)
    except:
        import traceback
        traceback.print_exc()
        bpm = 120
        offset = 0

    with open(path, "w", encoding='utf8') as f:
        # for line in meta.file_meta:
        #     if override is not None:
        #         for k, v in override.items():
        #             if line.startswith(k + ":"):
        #                 line = f"{k}: {v}"
        #                 break
        #     f.write(line + "\n")

        # if gridify is not None:
        #     f.write(f"[TimingPoints]\n{offset},{60000 / bpm},4,2,1,20,1,0\n\n")
        f.write(f"&first={offset / 1000}\n")
        f.write(f"&inote_5=({bpm})\n")

        prev = Beats()
        prev_content = ''
        curr_divide = None
        first = True
        for obj in hit_objects:
            interval = obj.timeStampInBeats - prev
            if interval.count > 0:
                if interval.divide != curr_divide:
                    f.write(f"{'{'}{interval.divide}{'}'}")
                    curr_divide = interval.divide
                f.write(f'{prev_content}')
                f.write(',' * interval.count)
                prev = obj.timeStampInBeats
                prev_content = obj.get_note_content()
            else:
                if first:
                    prev = obj.timeStampInBeats
                    prev_content = obj.get_note_content()
                prev_content += f'/{obj.get_note_content()}'

            first = False
            
        f.write(prev_content)
        f.write(',E')

    print(f"Succesfully saved to {path}")

class BaseMaimaiConvertor(metaclass=ABCMeta):

    def read_time(self, time):
        t = time * 1000 / self.rate + self.offset_ms
        index = int(t / self.frame_ms)
        offset = (t - index * self.frame_ms) / self.frame_ms
        return int(round(t)), index, offset

    def __init__(self, frame_ms, max_frame, mirror=False, from_logits=False, offset_ms=0,
                rate=1.0):
        self.frame_ms = frame_ms
        self.max_frame = max_frame
        self.mirror = mirror
        self.from_logits = from_logits
        self.offset_ms = offset_ms
        self.rate = rate


    @abstractmethod
    def objects_to_array(self, hit_objects: List[HitObject]) -> Tuple[np.ndarray, np.ndarray]: pass

    @abstractmethod
    def array_to_objects(self, note_array: np.ndarray) -> List[HitObject]: pass


    def timing_to_array(self, meta: BeatmapMeta) -> Tuple[np.ndarray, bool]:
        if len(meta.timing_points) == 0:
            return [None, False]

        red_lines = [] # (st, bpm)
        segment_list = [] # (st, visual_bpm, true_bpm)
        last_true_bpm = None

        for line in meta.timing_points:
            time_ms, timing = line.split(",")[:2]
            timing = float(timing)
            time_ms = float(time_ms)
            if timing < 0: # green line
                true_bpm = last_true_bpm * 100 / -timing
            else: # red lines
                true_bpm = 60000 / timing
                last_true_bpm = true_bpm
                if len(red_lines) == 0 or red_lines[-1][1] != true_bpm:
                    red_lines.append((time_ms, true_bpm))
            segment_list.append((time_ms, true_bpm, last_true_bpm))

        # detech visual sv
        cur_bpm = None
        has_sv = False
        if len(red_lines) > 1:
            for i in range(len(segment_list) - 1):
                if abs(segment_list[i][0] - segment_list[i + 1][0]) <= 1:
                    continue
                if cur_bpm is None:
                    cur_bpm = segment_list[i][1]
                else:
                    if abs(cur_bpm - segment_list[i][1]) > 0.00001:
                        has_sv = True
                        break

        # generate beat array
        array_length = min(self.max_frame, int(self.max_frame / self.rate))
        array = np.zeros((array_length, 2), dtype=np.float32)
        for i, (start_time_ms, true_bpm, _) in enumerate(segment_list):

            while true_bpm < 150:
                true_bpm = true_bpm * 2
            while true_bpm >= 300:
                true_bpm = true_bpm / 2
    
            if i == len(segment_list) - 1:
                end_time_ms = self.frame_ms * self.max_frame
            else:
                end_time_ms = segment_list[i + 1][0]
            beat_ms = start_time_ms
            while beat_ms <= end_time_ms:
                _, idx, offset = self.read_time(beat_ms)
                if idx >= array_length:
                    continue
                array[idx, 0] = 1
                array[idx, 1] = offset
                beat_ms += 60000 / true_bpm / 2
        
        return array, has_sv

class MaimaiConvertor(BaseMaimaiConvertor):
    def is_binary_positive(self, input):
        if self.from_logits:
            return input > 0
        else:
            return input > 0.5

    """
    Feature Layout:
        [is_start: 0/1] * key_count

        [offset_start: 0-1] * key_count
        valid only if is_start = 1

        [is_holding: 0/1] * key_count, (exclude start, include end),
        valid only if previous.is_start = 1 or previous.is_holding = 1

        [offset_end: 0-1]
        valid only if is_holding = 1 and latter.is_holding = 0
    """

    def array_to_objects(self, note_array: np.ndarray) -> List[HitObject]:
        note_array = note_array.transpose()
        # 定义每个数组的长度
        lengths = {
            'tap': 8,
            'tap_offset': 8,
            'is_holding': 8,
            'hold_end_offset': 8,
            'is_break': 8,
            'is_ex': 8,
            'is_slide_head': 8,
            'touch': 33,
            'touch_offset': 33,
            'is_hanabi': 33,
            'touch_holding': 1,
            'touch_hold_end_offset': 1,
            'slide_pass_through': 17,
            'slide_end_offset': 8
        }
        # 计算分割点
        split_indices = np.cumsum(list(lengths.values()))[:-1]
        # 使用np.split分割数组
        arrays = np.split(note_array, split_indices, 1)
        # 将分割后的数组与名称对应
        note_dict = dict(zip(lengths.keys(), arrays))
        tap = note_dict["tap"]
        tap_offset = note_dict["tap_offset"]
        is_holding = note_dict["is_holding"]
        hold_end_offset = note_dict["hold_end_offset"]
        is_break = note_dict["is_break"]
        is_ex = note_dict["is_ex"]
        is_slide_head = note_dict["is_slide_head"]
        touch = note_dict["touch"]
        touch_offset = note_dict["touch_offset"]
        is_hanabi = note_dict["is_hanabi"]
        touch_holding = note_dict["touch_holding"]
        touch_hold_end_offset = note_dict["touch_hold_end_offset"]
        slide_pass_through = note_dict["slide_pass_through"]
        slide_end_offset = note_dict["slide_end_offset"]

        hit_object_with_start = []
        key_count = 8
        for position in range(key_count):
            start_indices = np.where(self.is_binary_positive(tap[:, position]))[0]
            for start_index in start_indices:
                start_offset = np.clip(tap_offset[start_index, position], 0, 1)
                start = int(round((start_index + start_offset) * self.frame_ms))
                end = -1

                if start_index != len(note_array) - 1:
                    i = start_index + 1
                    while (i < len(note_array)
                           and self.is_binary_positive(is_holding[i, position])
                           and not self.is_binary_positive(tap[i, position])):
                        i += 1
                    end_index = i - 1
                    if end_index == start_index:
                        end = -1
                    else:
                        end_offset = np.clip(hold_end_offset[end_index, position], 0, 1)
                        end = int(round((end_index + end_offset) * self.frame_ms))

                obj = HitObject()
                if end == -1:
                    obj.timeStamp = start
                    obj.startPosition = position + 1
                    obj.noteType = 0
                else:
                    obj.timeStamp = start
                    obj.startPosition = position + 1
                    obj.noteType = 2
                    obj.holdTime = (end - start)
                obj.isBreak = self.is_binary_positive(is_break[start_index, position])
                obj.isEx = self.is_binary_positive(is_ex[start_index, position])
                hit_object_with_start.append((obj, start))

        touch_count = 33
        for touch_id in range(touch_count):
            start_indices = np.where(self.is_binary_positive(touch[:, touch_id]))[0]
            for start_index in start_indices:
                start_offset = np.clip(touch_offset[start_index, touch_id], 0, 1)
                start = int(round((start_index + start_offset) * self.frame_ms))
                end = -1
                
                if touch_id == Touch('C').get_id():
                    if start_index != len(note_array) - 1:
                        i = start_index + 1
                        while (i < len(note_array)
                            and self.is_binary_positive(touch_holding[i, 0])
                            and not self.is_binary_positive(touch[i, touch_id])):
                            i += 1
                        end_index = i - 1
                        if end_index == start_index:
                            end = -1
                        else:
                            end_offset = np.clip(touch_hold_end_offset[end_index, 0], 0, 1)
                            end = int(round((end_index + end_offset) * self.frame_ms))

                obj = HitObject()
                if end == -1:
                    obj.timeStamp = start
                    obj.touchArea = Touch.from_id(touch_id)
                    obj.noteType = 3
                else:
                    obj.timeStamp = start
                    obj.touchArea = Touch.from_id(touch_id)
                    obj.noteType = 4
                    obj.holdTime = (end - start)
                obj.isHanabi = self.is_binary_positive(is_hanabi[start_index, touch_id])
                hit_object_with_start.append((obj, start))
        hit_object_with_start = sorted(hit_object_with_start, key=lambda x: x[1])
        return list(map(lambda x: x[0], hit_object_with_start))

    def objects_to_array(self, hit_objects: dict) -> Tuple[np.ndarray, np.ndarray]:
        array_length = min(self.max_frame, int(self.max_frame / self.rate))
        tap = np.zeros((array_length, 8), dtype=np.float32)
        tap_offset = np.zeros((array_length, 8), dtype=np.float32)
        is_holding = np.zeros((array_length, 8), dtype=np.float32)
        hold_end_offset = np.zeros((array_length, 8), dtype=np.float32)
        is_break = np.zeros((array_length, 8), dtype=np.float32)
        is_ex = np.zeros((array_length, 8), dtype=np.float32)
        is_slide_head = np.zeros((array_length, 8), dtype=np.float32)
        touch = np.zeros((array_length, 33), dtype=np.float32)
        touch_offset = np.zeros((array_length, 33), dtype=np.float32)
        is_hanabi = np.zeros((array_length, 33), dtype=np.float32)
        touch_holding = np.zeros((array_length, 1), dtype=np.float32)
        touch_hold_end_offset = np.zeros((array_length, 1), dtype=np.float32)
        slide_pass_through = np.zeros((array_length, 17), dtype=np.float32)
        slide_end_offset = np.zeros((array_length, 8), dtype=np.float32)
        max_index = 0

        position_map = list(range(1, 8 + 1))
        if self.mirror:
            position_map = [9 - position_map[i] for i in range(8)]

        for timestamp, notes in hit_objects.items():
            timestamp = float(timestamp)
            start, start_index, start_offset = self.read_time(timestamp)
            # is_start / offset_start

            if start_index >= array_length:
                continue

            for note in notes:
                position = note.get('startPosition', 1)
                position = position_map[position - 1]
                position -= 1

                noteType = note.get("noteType", 0)
                
                # tap
                if noteType == 0:
                    tap[start_index, position] = 1
                    tap_offset[start_index, position] = start_offset
                # slide
                elif noteType == 1:
                    if not note["isSlideNoHead"]:
                        tap[start_index, position] = 1
                        tap_offset[start_index, position] = start_offset
                        is_slide_head[start_index, position] = 1

                    slide_path = get_slide_path(note.get('noteContent', ''))
                    
                    # 在slide_pass_through里平按顺序均分配Touch的时间值
                    slide_start_time = note.get("slideStartTime", 0)
                    slide_duration = note.get("slideTime", 0)
                    end, end_index, end_offset = self.read_time(slide_start_time + slide_duration)

                    if slide_path and slide_duration > 0:
                        # 计算每个路径点之间的时间间隔
                        time_per_point = slide_duration / len(slide_path) if len(slide_path) > 1 else 0
                        
                        enter_time = slide_start_time
                        # 处理每个路径点
                        for i, touches in enumerate(slide_path):
                            # 计算这个路径点的时间戳
                            leave_time = enter_time + time_per_point
                            
                            # 获取这个时间点在array中的索引和偏移
                            _, enter_index, _ = self.read_time(enter_time)
                            _, leave_index, _ = self.read_time(leave_time)

                            if leave_index >= array_length:
                                leave_index = array_length - 1
                            
                            if isinstance(touches, Touch):
                                touches = (touches,)
                            for _touch in touches:
                                # 获取这个触摸点的ID
                                _touch = Touch(_touch.area, position_map[_touch.position - 1])
                                touch_id = _touch.get_id()
                                
                                # 在slide_pass_through中标记这个路径点
                                for pass_index in range(enter_index, leave_index + 1):
                                    slide_pass_through[pass_index, touch_id] = 1

                            if leave_index == array_length - 1:
                                break
                            
                            enter_time = leave_time
                            
                        # 处理结束点的偏移
                        if end_index < array_length:
                            last_touches = slide_path[-1]
                            if isinstance(last_touches, Touch):
                                last_touches = (last_touches,)
                            for last_touch in last_touches:
                                last_touch = Touch(last_touch.area, position_map[last_touch.position - 1])
                                slide_end_offset[end_index, last_touch.position - 1] = end_offset
                        
                        end_index = min(end_index, array_length - 1)
                        max_index = max(end_index, max_index)
                # hold
                elif noteType == 2:
                    tap[start_index, position] = 1
                    tap_offset[start_index, position] = start_offset

                    end, end_index, end_offset = self.read_time(timestamp + note.get("holdTime", 0))
                    if end_index >= array_length:
                        end_index = array_length - 1
                        end_offset = 1
                    for i in range(start_index + 1, end_index + 1):
                        is_holding[i, position] = 1
                    hold_end_offset[end_index, position] = end_offset
                    max_index = max(end_index, max_index)
                # touch
                elif noteType == 3:
                    touch_area = note.get("touchArea", 'C')
                    touch_id = Touch(touch_area, position + 1).get_id()
                    touch[start_index, touch_id] = 1
                    touch_offset[start_index, touch_id] = start_offset

                    if note.get('isHanabi', False):
                        is_hanabi[start_index, touch_id] = 1
                # touch hold
                elif noteType == 4:
                    assert note.get("touchArea", '') == 'C', "Touch Hold must be in C Area"
                    touch_id = Touch('C').get_id()
                    
                    touch[start_index, touch_id] = 1
                    touch_offset[start_index, touch_id] = start_offset

                    if note.get('isHanabi', False):
                        is_hanabi[start_index, touch_id] = 1

                    end, end_index, end_offset = self.read_time(timestamp + note.get("holdTime", 0))
                    if end_index >= array_length:
                        end_index = array_length - 1
                        end_offset = 1
                    for i in range(start_index + 1, end_index + 1):
                        touch_holding[i, 0] = 1
                    touch_hold_end_offset[end_index, 0] = end_offset
                    max_index = max(end_index, max_index)

                if note.get('isBreak', False):
                    is_break[start_index, position] = 1
                if note.get('isEx', False):
                    is_ex[start_index, position] = 1

            max_index = max(start_index, max_index)

        array = np.hstack([
            tap,                    # 8
            tap_offset,            # 8
            is_holding,            # 8
            hold_end_offset,       # 8
            is_break,              # 8
            is_ex,                 # 8
            is_slide_head,         # 8
            touch,               # 33
            touch_offset,          # 33
            is_hanabi,             # 33
            touch_holding,         # 1
            touch_hold_end_offset, # 1
            slide_pass_through,    # 17
            slide_end_offset       # 8
        ])

        if len(array) < self.max_frame:
            array = np.concatenate([
                array,
                np.zeros((self.max_frame - len(array), array.shape[1]), dtype=np.float32)
            ], axis=0)
        valid_flag = np.zeros((len(array),))
        valid_flag[:max_index] = 1
        array = np.transpose(array)
        return array, valid_flag


if __name__ == "__main__":
    # map_path = """E:\E\osu!\Songs\891164 Various Artists - 4K LN Dan Courses v2 - Extra Level -\Various Artists - 4K LN Dan Courses v2 - Extra Level - (_underjoy) [13th Dan - Yoru (Marathon)].osu"""
    # map_path = r"""E:\E\osu!\Songs\1395676 goreshit - thinking of you\goreshit - thinking of you (hna) [obsession 1.1x (250bpm)].osu"""
    song_data = {'name': 'QuiQ', 'style': 'DX', 'diff': 'MASTER', 'cc': '14.4', 'path': '19. UNiVERSE PLUS/World_s end loneliness'}
    path = os.path.join('/Volumes/XelesteSSD/maiCharts/json', song_data['path'])
    chart_path = os.path.join(path, f"{song_data['diff']}.json")
    audio_path = os.path.join(path, "track.mp3")
    objs, beatmap_meta = get_maimai_data(chart_path, audio_path, song_data, convertor_params={"frame_ms": 2048 / 22050 / 2 * 1000,
                                                                    "max_frame": 8192,
                                                                    "mirror": False,
                                                                    "offset_ms": 0,
                                                                    "rate": 1.0,
                                                                    })
    
    save_maimai_file(beatmap_meta,
                  beatmap_meta.convertor.objects_to_array(objs)[0],
                  os.path.join("/Volumes/XelesteSSD", "maidata.txt"),
                  gridify)