import yaml
import os
import torch
from mai.util import instantiate_from_config
from mai.data.convertor import get_maimai_data, save_maimai_file

# Load the YAML configuration file
with open('configs/mai/autoencoder_cli_tap.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Instantiate the model
model_config = config['model']
model = instantiate_from_config(model_config)

# Load model from checkpoint
ckpt_path = 'models/ckpt/model_TAP_epoch=038.ckpt'
model.init_from_ckpt(ckpt_path)

# Set model to evaluation mode
model.eval()

# Load single chart file
chart_path = "path/to/your/chart.mai"  # Replace with your chart path
song_data = None  # Optional song metadata
data, meta = get_maimai_data(chart_path, None, song_data, None)

# Convert to tensor and add batch dimension
notes = torch.tensor(data['note']).unsqueeze(0)

# Perform inference
with torch.no_grad():
    reconstructions, z = model(notes)
    
    # Save reconstructed chart
    output_dir = 'reconstructions'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reconstruction.mai')
    
    save_maimai_file(meta, reconstructions[0].cpu().numpy(), output_path)
    
    print(f"Input file: {chart_path}")
    print(f"Saved reconstruction to: {output_path}")
    print("Latent Variables shape:", z.shape)