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
_logger = logging.getLogger("token_classif_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/token_classif_wikiann.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models

hf_api = HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="token-classification", language='en'),
    sort="downloads", direction=-1, limit=20)
token_classif_models = [l.modelId for l in models if 'ontonotes' not in l.modelId and 'flair' not in l.modelId and 'Bio' not in l.modelId][1:9]

def dset_gen():
    dset = load_dataset("wikiann","en", streaming=True, split="test")
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

for model in token_classif_models:
    print(model)
    tracker = EmissionsTracker(project_name=model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/token_classif_wikiann.csv')
    tracker.start()
    tracker.start_task("load model")
    classifier = pipeline('token-classification', model=model, device=0)
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count =0
    for sent in dset:
        count+=1
        classifier(' '.join(sent['tokens']))
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
