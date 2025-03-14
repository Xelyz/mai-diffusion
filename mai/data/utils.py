import re
import math
from mug.data.utils import timing
import numpy as np

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

    def __str__(self):
        return str({
            "timeStamp": self.timeStamp,
            "timeStampInBeats": self.timeStampInBeats,
            "holdTime": self.holdTime,
            "holdTimeInBeats": self.holdTimeInBeats,
            "isBreak": self.isBreak,
            "isEx": self.isEx,
            "isHanabi": self.isHanabi,
            "isSlideNoHead": self.isSlideNoHead,
            "noteType": self.noteType,
            "slideStartTime": self.slideStartTime,
            "slideTime": self.slideTime,
            "slideShape": self.slideShape,
            "startPosition": self.startPosition,
            "touchArea": str(self.touchArea)
        })

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