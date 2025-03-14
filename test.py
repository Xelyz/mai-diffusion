import yaml
import os
import torch
from mai.util import instantiate_from_config
from mai.data.convertor import get_maimai_data, save_maimai_file
from mai.data.utils import gridify

# Define converter parameters
converter_params = {
    "frame_ms": 512 // 4 / 22050 * 8 * 1000,
    "max_frame": 32768 // 8,
    'mirror': False,  # whether to mirror the chart
    'from_logits': True,  # whether the model outputs logits
    'offset_ms': 0,  # offset in milliseconds
    'rate': 1.0  # playback rate
}

# Load single chart file
chart_path = "/Volumes/XelesteSSD/maiCharts/json/Cryst/Abstruse Dilemma/MASTER.json"  # Replace with your chart path
song_data = None  # Optional song metadata
data, meta = get_maimai_data(chart_path, None, song_data, converter_params)
obj_array, valid_flag = meta.convertor.objects_to_array(data, "tap")

obj_array = torch.tensor(obj_array)

reconstructions = torch.nn.functional.pad(
            obj_array, 
            (0, 0, 0, 125 - obj_array.shape[0])
        )

print(reconstructions.shape)

# Save reconstructed chart
output_dir = 'reconstructions'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'copy.txt')

save_maimai_file(meta, reconstructions.numpy(), output_path)

print(f"Input file: {chart_path}")
print(f"Saved reconstruction to: {output_path}")