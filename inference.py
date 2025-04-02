import yaml
import os
import torch
from mai.util import instantiate_from_config
from mai.data.convertor import get_maimai_data, save_maimai_file

# Load the YAML configuration file
with open('configs/mai/autoencoder_cli.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Instantiate the model
model_config = config['model']
model = instantiate_from_config(model_config)

# Load model from checkpoint
ckpt_path = 'models/ckpt/model_TAP_epoch=056.ckpt'
model.init_from_ckpt(ckpt_path)

# Set model to evaluation mode
model.eval()

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
chart_path = "/Volumes/XelesteSSD/maiCharts/json/Cryst/Singularity/MASTER.json"  # Replace with your chart path
song_data = None  # Optional song metadata
data, meta = get_maimai_data(chart_path, None, song_data, converter_params)
obj_array, valid_flag = meta.convertor.objects_to_array(data, None)

# Convert to tensor and add batch dimension
notes = torch.tensor(obj_array).unsqueeze(0)

# Perform inference
with torch.no_grad():
    reconstructions, z = model(notes)

    # Save reconstructed chart
    output_dir = 'reconstructions'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reconstruction.txt')
    
    save_maimai_file(meta, reconstructions[0].cpu().numpy(), output_path)
    
    print(f"Input file: {chart_path}")
    print(f"Saved reconstruction to: {output_path}")