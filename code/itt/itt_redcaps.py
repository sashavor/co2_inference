from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer
from PIL import Image
from io import BytesIO
import requests

from codecarbon import EmissionsTracker
import huggingface_hub
from huggingface_hub import HfApi, ModelFilter
import logging
import torch
import os


# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("itt_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/itt_testing_redcaps.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load TTI models
hf_api=HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="image-to-text"),
    sort="downloads", direction=-1, limit=8)
itt_model = [l.modelId for l in models]

def dset_gen():
    dset = load_dataset("red_caps", "roses_2020", split= 'train', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        im = Image.open(BytesIO(requests.get(row['image_url']).content))
        row['image'] = im.resize((640, 480), resample=0).convert('RGB')
        yield row

dset = Dataset.from_generator(dset_gen)

for itt_model in itt_models:
    print(itt_model)
    tracker = EmissionsTracker(project_name=itt_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/itt_redcaps_git.csv')
    tracker.start()
    tracker.start_task("load model")
    image_to_text = pipeline("image-to-text", model=itt_model, device=0 )
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count = 0
    for d in dset:
        image_to_text(d['image'])
        count+=1
    print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()
