import time
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import math

class Beats:
    def __init__(self, divide = 4, count = 0):
        self.divide = divide
        self.count = count

    def value(self):
        return self.count / self.divide
    
    def reduce(self):
        gcd = math.gcd(self.count, self.divide)
        self.count //= gcd
        self.divide //= gcd
    
    def __repr__(self):
        return f'divide: {self.divide} / count: {self.count}'
    
    def __sub__(self, other):
        if isinstance(other, Beats):
            if self.divide == other.divide:
                return Beats(self.divide, self.count - other.count)
            else:
                res = Beats()
                res.divide = math.lcm(self.divide, other.divide)
                res.count = res.divide // self.divide * self.count - res.divide // other.divide * other.count
                res.reduce()
                return res
        else:
            raise NotImplementedError

class Touch:
    def __init__(self, area, position = 1):
        self.area = area
        self.position = int(position) # 1-8

    def get_id(self):
        if self.area == 'C':
            return 16  # C只有1个区域，固定返回16
        elif self.area in "ABDE":
            return "ABDE".index(self.area) * 8 + (self.position - 1) + (1 if self.area in "DE" else 0)
        
    @staticmethod
    def from_id(id):
        if id == 16:
            return Touch('C')
        else:
            if id > 16:
                id -= 1
            area = "ABDE"[id // 8]
            position = id % 8 + 1
            return Touch(area, position)

    def __repr__(self):
        if self.area == 'C':
            return f'{self.area}'
        return f'{self.area}{self.position}'

    def __add__(self, other):
        return Touch(self.area, self.position + other)
    
    def __sub__(self, other):
        return Touch(self.area, self.position - other)

    def __mul__(self, other):
        return Touch(self.area, self.position * other)

    def __mod__(self, other):
        return Touch(self.area, self.position % other)

class HitObject:
    timeStamp: float = 0.0
    timeStampInBeats: Beats = Beats(4, 0)
    holdTime: float = 0.0
    holdTimeInBeats: Beats = Beats(4, 0)
    isBreak: bool = False
    isEx: bool = False
    isHanabi: bool = False
    isSlideNoHead: bool = False
    noteType: int = 0
    slideStartTime: float = 0.0
    slideTime: float = 0.0
    slideShape: str = " "
    startPosition: int = 0
    touchArea: Touch = Touch('C')

    def get_note_content(self):
        if self.noteType in [0, 1, 2]:
            content = str(self.startPosition)
            if self.isBreak:
                content += "b"
            if self.isEx:
                content += "x"
        elif self.noteType in [3, 4]:
            content = str(self.touchArea)
            if self.isHanabi:
                content += "f"

        if self.noteType in [2, 4]:
            content += "h"
            content += f"[{self.holdTimeInBeats.divide}:{self.holdTimeInBeats.count}]"

        return content
    
def parse_hit_objects(obj: HitObject):
    if obj is None:
        return None, None, None
    return obj.timeStamp, obj.startPosition, obj.holdTime

def test_timing(time_list, test_bpm, test_offset, div, refine):
    cur_offset = test_offset
    cur_bpm = test_bpm

    epsilon = 10
    gap = 60 * 1000 / (test_bpm * div)
    delta_time_list = time_list - test_offset
    meter_list = delta_time_list / gap
    meter_list_round = np.round(meter_list)
    timing_error = np.abs(meter_list - meter_list_round)
    valid = (timing_error < epsilon / gap).astype(np.int32)
    valid_count = np.sum(valid)

    if valid_count >= 2 and refine:
        rgs = LinearRegression(fit_intercept=True)
        rgs.fit(meter_list_round.reshape((-1, 1)), time_list, sample_weight=valid)
        if not np.isinf(rgs.coef_) and not np.isnan(rgs.coef_) and rgs.coef_[0] != 0:
            cur_offset = rgs.intercept_
            cur_bpm = 60000 / rgs.coef_[0] / 4

            while cur_bpm < 150:
                cur_bpm = cur_bpm * 2
            while cur_bpm >= 300:
                cur_bpm = cur_bpm / 2

    # valid_ratio = valid_count
    valid_ratio = valid_count / test_bpm
    return valid_ratio, valid, cur_bpm, cur_offset


def timing(time_list, verbose=True):
    offset = time_list[0]

    best_bpm = None
    best_offset = None
    best_valid_ratio = -1

    # find the best bpm when offset = first time
    st = time.time()
    for test_bpm in np.arange(150, 300, 0.1):\

        valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, test_bpm, offset, div=1,
                                                              refine=False)

        if valid_ratio > best_valid_ratio:
            valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, test_bpm, offset,
                                                                  div=1,
                                                                  refine=True)
            best_valid_ratio = valid_ratio
            best_bpm = cur_bpm
            best_offset = cur_offset
            if verbose:
                print(f"[valid: {valid_ratio} / {len(valid)}] bpm {test_bpm} -> {cur_bpm}, "
                f"offset {offset} -> {cur_offset}")

        # find the best offset when bpm = best bpm
        gap = 60000 / cur_bpm
        for test_offset in np.arange(best_offset, best_offset - gap, -gap / 4):

            valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, cur_bpm,
                                                                  test_offset,
                                                                  div=1,
                                                                  refine=False)
            if valid_ratio > best_valid_ratio:
                valid_ratio, valid, cur_bpm, cur_offset = test_timing(time_list, cur_bpm,
                                                                      test_offset,
                                                                      div=1,
                                                                      refine=True)
                best_valid_ratio = valid_ratio
                best_bpm = cur_bpm
                best_offset = cur_offset
                if verbose:
                    print(f"[valid: {valid_ratio} / {len(valid)}] bpm {best_bpm} -> {cur_bpm}, "
                    f"offset {offset} -> {cur_offset}")

    _, valid_8, best_bpm, best_offset = test_timing(time_list, best_bpm, best_offset, div=16,
                                                    refine=False)
    _, valid_6, best_bpm, best_offset = test_timing(time_list, best_bpm, best_offset, div=6,
                                                    refine=False)
    valid = np.clip(valid_6 + valid_8, 0, 1)

    if verbose:
        print("Test time:", time.time() - st)
        print(f"Final bpm: {best_bpm}, offset: {best_offset}")
        print(f"Final valid: {np.sum(valid)} / {len(valid)}")
        print(f"Invalid: {time_list[valid == 0]}")

    return best_bpm, best_offset

    # rgs = LinearRegression(fit_intercept=True)
    # rgs.fit(np.asarray(meters).reshape((-1, 1)), times[:i + 1])

epsilon = 10

def gridify(hit_objects: list[HitObject], verbose=True):
    times = []
    for obj in hit_objects:
        st, _, _ = parse_hit_objects(obj)
        times.append(st)
    times = np.asarray(times, dtype=np.float32)
    bpm, offset = timing(times, verbose)

    def format_time(t, _offset):
        for div in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
            gap = 60 * 1000 / (bpm * div / 4)
            meter = (t - _offset) / gap
            meter_round = round(meter)
            timing_error = abs(meter - meter_round)
            if timing_error < epsilon / gap:
                return str(int(meter_round * gap + _offset)), Beats(div, meter_round)
        div = 256
        gap = 60 * 1000 / (bpm * div / 4)
        meter = (t - _offset) / gap
        meter_round = round(meter)
        return int(t), Beats(div, meter_round)

    new_hit_objects = []
    for obj in hit_objects:
        obj.timeStamp, obj.timeStampInBeats = format_time(obj.timeStamp, offset)
        if obj.noteType in [2, 4]:
            obj.holdTime, obj.holdTimeInBeats = format_time(obj.holdTime, 0)
        new_hit_objects.append(obj)
    return bpm, offset, new_hit_objects
        
def get_slide_path(note_content: str):
    i = 0
    path = []
    position = None
    slide_shape = None
    note_content = re.sub(r'\[.*?\]', '', note_content)
    while i < len(note_content):
        char = note_content[i]
        if not position:
            if char.isdigit():
                position = int(char)
        elif not slide_shape:
            if char in "<>^-vVszpqw":
                slide_shape = char
        else:
            if char in 'pq' and char == slide_shape:
                slide_shape += char
            elif char.isdigit():
                path.extend(get_slide_component_path(position, slide_shape, int(char)))
                position = int(char)
                if slide_shape == 'w':
                    return path
                slide_shape = None if slide_shape != 'V' else '-'
        i += 1
    path.append(Touch("A", position))
    return path

slide_shape_mapping = {
    "l2": [Touch('A', 0), Touch('B', 1)],
    "l3": [Touch('A', 0), Touch('B', 1), Touch('B', 2)],
    "l4": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 4)],
    "w": [Touch('A', 0), (Touch('B', 0), Touch('B', 1), Touch('B', 7)), (Touch('C'), Touch('B', 2), Touch('B', 6)), (Touch('B', 4), Touch('A', 3), Touch('A', 5))],
    "r1": [Touch('A', 0)],
    "r2": [Touch('A', 0), Touch('A', 1)],
    "r3": [Touch('A', 0), Touch('A', 1), Touch('A', 2)],
    "r4": [Touch('A', 0), Touch('A', 1), Touch('A', 2), Touch('A', 3)],
    "r5": [Touch('A', 0), Touch('A', 1), Touch('A', 2), Touch('A', 3), Touch('A', 4)],
    "r6": [Touch('A', 0), Touch('A', 1), Touch('A', 2), Touch('A', 3), Touch('A', 4), Touch('A', 5)],
    "r7": [Touch('A', 0), Touch('A', 1), Touch('A', 2), Touch('A', 3), Touch('A', 4), Touch('A', 5), Touch('A', 6)],
    "r0": [Touch('A', 0), Touch('A', 1), Touch('A', 2), Touch('A', 3), Touch('A', 4), Touch('A', 5), Touch('A', 6), Touch('A', 7)],
    "z": [Touch('A', 0), Touch('B', 1), Touch('B', 2), Touch('C'), Touch('B', 6), Touch('B', 5)],
    "vh": [Touch('A', 0), Touch('B', 0)],
    "vt": [Touch('C'), Touch('B', 0)],
    "p0": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3), Touch('B', 2), Touch('B', 1)],
    "p1": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3), Touch('B', 2)],
    "p2": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3)],
    "p3": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4)],
    "p4": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5)],
    "p5": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3), Touch('B', 2), Touch('B', 1), Touch('B', 0), Touch('B', 7), Touch('B', 6)],
    "p6": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3), Touch('B', 2), Touch('B', 1), Touch('B', 0), Touch('B', 7)],
    "p7": [Touch('A', 0), Touch('B', 7), Touch('B', 6), Touch('B', 5), Touch('B', 4), Touch('B', 3), Touch('B', 2), Touch('B', 1), Touch('B', 0)],
    "pp0": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1)],
    "pp1": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2)],
    "pp2": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3)],
    "pp3": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1), Touch('B', 0), Touch('C'), Touch('B', 3)],
    "pp4": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1), Touch('B', 0), Touch('C'), Touch('B', 4)],
    "pp5": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1), Touch('B', 0), Touch('C'), Touch('B', 5)],
    "pp6": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1), Touch('B', 0), Touch('B', 7)],
    "pp7": [Touch('A', 0), Touch('B', 0), Touch('C'), Touch('B', 3), Touch('A', 2), Touch('A', 1), Touch('B', 0)]
}

def get_slide_component_path(start, slide_shape, end):
    """
    exclude the A area at the end because of fes slides
    start, end: 1-8
    """
    reflect = 1
    distance = (end - start) % 8
    path = []
    match slide_shape:
        case '-' | 'V':
            if distance > 4:
                reflect = -1
                distance = 8 - distance
            assert distance != 1, f"{start}{slide_shape}{end}"
            path = slide_shape_mapping[f'l{distance}']
        case '^':
            if distance > 4:
                reflect = -1
                distance = 8 - distance
            path = slide_shape_mapping[f'r{distance}']
        case '<':
            if start in [1, 2, 7, 8]:
                reflect = -1
                distance = (8 - distance) % 8
            path = slide_shape_mapping[f'r{distance}']
        case '>':
            if start in [3, 4, 5, 6]:
                reflect = -1
                distance = (8 - distance) % 8
            path = slide_shape_mapping[f'r{distance}']
        case 'p':
            path = slide_shape_mapping[f'p{distance}']
        case 'q':
            reflect = -1
            distance = (8 - distance) % 8
            path = slide_shape_mapping[f'p{distance}']
        case 'pp':
            path = slide_shape_mapping[f'pp{distance}']
        case 'qq':
            reflect = -1
            distance = (8 - distance) % 8
            path = slide_shape_mapping[f'pp{distance}']
        case 'z':
            assert distance == 4, f"{start}{slide_shape}{end}"
            path = slide_shape_mapping['z']
        case 's':
            reflect = -1
            distance = 8 - distance
            assert distance == 4, f"{start}{slide_shape}{end}"
            path = slide_shape_mapping['z']
        case 'v':
            assert start != end, f"{start}{slide_shape}{end}"
            path.extend([(touch + start - 1) % 8 + 1 for touch in slide_shape_mapping['vh']])
            path.extend([(touch + end - 1) % 8 + 1 for touch in slide_shape_mapping['vt']])
            return path
        case 'w':
            distance %= 8
            assert distance == 4, f"{start}{slide_shape}{end}"

            path = slide_shape_mapping['w']
            new_path = []
            for touches in path:
                if isinstance(touches, Touch):
                    new_path.append((touches + start - 1) % 8 + 1)
                else:
                    touches = [(touch + start - 1) % 8 + 1 for touch in touches]
                    new_path.append(touches)
            return new_path
    return [(touch * reflect + start - 1) % 8 + 1 for touch in path]

def remove_intractable_mania_mini_jacks(hit_objects, verbose=True, jack_interval=90):
    key_count = 4  # TODO
    column_width = int(512 / key_count)
    new_hit_objects = [x for x in hit_objects]

    def has_ln(start_index, column, time):
        i = start_index - 1
        while i >= 0:
            start_time, c, end_time = parse_hit_objects(new_hit_objects[i], column_width)
            i -= 1
            if end_time is None or start_time is None:
                continue
            if c == column and start_time <= time:
                return end_time >= time - 50
        return False


    def get_notes_idx_in_interval(start_index, time, interval, column, search_previous,
                                  search_latter):
        result = []
        i = start_index - 1
        if search_previous:
            while i >= 0:
                st, c, _ = parse_hit_objects(new_hit_objects[i], column_width)
                if st is not None:
                    if abs(st - time) <= interval:
                        if c == column or column < 0:
                            result.append((i, st, c))
                    else:
                        break
                i -= 1
        if search_latter:
            i = start_index + 1
            while i < len(new_hit_objects):
                st, c, _ = parse_hit_objects(new_hit_objects[i], column_width)
                if st is not None:
                    if abs(st - time) <= interval:
                        if c == column or column < 0:
                            result.append((i, st, c))
                    else:
                        break
                i += 1
        return result

    for i in range(len(new_hit_objects)):
        start_time, column, end_time = parse_hit_objects(new_hit_objects[i], column_width)

        previous_jacks = get_notes_idx_in_interval(i, start_time, jack_interval, column,
                                                   search_previous=True, search_latter=False)
        if len(previous_jacks) != 0:
            # Detect jacks!
            # Step 1: judge if it's an end of streams. If so, ignore it.
            notes_after_it = get_notes_idx_in_interval(i, start_time, jack_interval * 2, -1,
                                                       search_previous=False,
                                                       search_latter=True)
            count_notes_after_it = 0
            for n in notes_after_it:
                if abs(n[1] - start_time) >= epsilon:
                    count_notes_after_it += 1
            if count_notes_after_it == 0:
                if verbose:
                    print(f"Ignore: {start_time}, {column}")
                continue

            # Step 2: try to move the notes to other columns.
            # Priority: latter note > previous note, same side > other sides
            success = False
            for (is_ln, try_move_index, try_move_t, try_move_src_column) in [
                (end_time is not None, i, start_time, column),
                (False, ) + previous_jacks[0]
            ]:
                if is_ln:
                    continue # we don't want to move LN since it's intractable
                if try_move_src_column == 0 or try_move_src_column == 1:
                    try_move_dst_columns = (1 - try_move_src_column, 2, 3)
                else:
                    try_move_dst_columns = (5 - try_move_src_column, 1, 0)

                for try_move_dst_column in try_move_dst_columns:
                    if has_ln(try_move_index, try_move_dst_column, try_move_t):
                        continue
                    jacks_after_move = len(get_notes_idx_in_interval(
                        try_move_index, try_move_t, jack_interval, try_move_dst_column,
                        search_previous=True, search_latter=True
                    ))
                    if jacks_after_move == 0:
                        success = True
                        if verbose:
                            print(f"Move: {try_move_t}, {try_move_src_column} -> {try_move_dst_column}")

                        elements = new_hit_objects[try_move_index].split(",")
                        elements[0] = str(int(round((try_move_dst_column + 0.5) * column_width)))
                        new_hit_objects[try_move_index] = ",".join(elements)

                        break
                if success:
                    break
            if success:
                continue

            # Step 3: Remove the note that has the more holds
            holds_latter = len(
                get_notes_idx_in_interval(i, start_time, 10, -1, search_previous=True,
                                          search_latter=True)
            ) + 1
            holds_previous = len(
                get_notes_idx_in_interval(previous_jacks[0][0], previous_jacks[0][1],
                                          10, -1, search_previous=True,
                                          search_latter=True)
            ) + 1
            if holds_latter > 1 and holds_latter >= holds_previous and end_time is None:
                if verbose:
                    print(f"Remove: {start_time} | {column} "
                          f"due to the holds: {holds_latter} >= {holds_previous}")
                new_hit_objects[i] = None
            elif holds_previous > 1 and holds_previous >= holds_latter:
                if verbose:
                    print(f"Remove: {previous_jacks[0][1]} | {column} "
                          f"due to the holds: {holds_latter} >= {holds_previous}")
                new_hit_objects[previous_jacks[0][0]] = None
            elif end_time is not None: # LN, remove previous
                if verbose:
                    print(f"Remove: {previous_jacks[0][1]} | {column} "
                          f"due to LN")
                new_hit_objects[previous_jacks[0][0]] = None
            else:
                if verbose:
                    print(f"Remove: {start_time} | {column} "
                          f"for no reason")
                new_hit_objects[i] = None

    return [x for x in new_hit_objects if x is not None]


if __name__ == "__main__":
    # print(','.join(map(str, get_slide_path('4-2w6'))))
    print(Beats(4, 5) - Beats(8, 3))

    # from mug.data.convertor import parse_osu_file, save_osu_file
    # import sys

    # path = sys.argv[-1]

    # new_path = path.replace(".osu", "_refine.osu")

    # hit_objects, meta = parse_osu_file(
    #     path,
    #     None)

    # new_hit_objects = remove_intractable_mania_mini_jacks(hit_objects)

    # override = {
    #     "Version": "rm jack"
    # }

    # with open(new_path, "w", encoding='utf8') as f:
    #     for line in meta.file_meta:
    #         if override is not None:
    #             for k, v in override.items():
    #                 if line.startswith(k + ":"):
    #                     line = f"{k}: {v}"
    #                     break
    #         f.write(line + "\n")

    #     f.write("[HitObjects]\n")

    #     for hit_object in new_hit_objects:
    #         f.write(hit_object + "\n")