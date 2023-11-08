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
_logger = logging.getLogger("text_classif_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/text_classif_tomatoes.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models

hf_api= HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
        language="en",
		task="text-classification"),
    sort="downloads", direction=-1, limit=20)
text_classif_models = [l.modelId for l in models if 'fin' not in l.modelId][:8]


def dset_gen():
    dset = load_dataset("rotten_tomatoes", split= 'validation', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

for model in text_classif_models:
    print(model)
    tracker = EmissionsTracker(project_name=model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/text_classif_tomatoes.csv')
    tracker.start()
    tracker.start_task("load model")
    classifier = pipeline("text-classification", model=model, device=0 )
    model_emissions = tracker.stop_task()
    count=0
    for d in dset:
        count+=1
        classifier(d['text'])
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
