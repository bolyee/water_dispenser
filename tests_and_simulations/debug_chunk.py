import traceback
import sys
import torch
sys.path.append('.')
from demo.util import load_model, get_model_output, load_audio_tensor

model = load_model()
a = load_audio_tensor('media_assets/example_video.mp4')
print("Full shape:", a.shape)

chunk = a.squeeze()[8000:32000]
print("Chunk shape:", chunk.shape)

try:
    with torch.no_grad():
         get_model_output(chunk, model)
    print("Success!")
except Exception as e:
    traceback.print_exc()
