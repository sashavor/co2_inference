from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer

from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, ModelFilter
import logging
import torch
import huggingface_hub
import einops

import os


# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("obj_detect_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/obj_detect_testing_coco.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models

hf_api= HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="object-detection"),
    sort="downloads", direction=-1, limit=30)
obj_detect_models = [l.modelId for l in models if 'table' not in l.modelId and 'license-plate' not in l.modelId and 'yolov8m' not in l.modelId and 'owlvit' not in l.modelId and 'yolov8' not in l.modelId][:8]

def dset_gen():
    dset = load_dataset("rafaelpadilla/coco2017", split= 'val', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        im = row['image'].resize((640, 480), resample=0).convert('RGB')
        row['image'] = im
        yield row

dset = Dataset.from_generator(dset_gen)

for model in obj_detect_models:
    print(model)
    tracker = EmissionsTracker(project_name=model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/obj_detection_coco_resized.csv')
    tracker.start()
    tracker.start_task("load model")
    classifier = pipeline("object-detection", model=model, device=0 )
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count=0
    for d in dset:
        count+=1
        classifier(d['image'])
    print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()
