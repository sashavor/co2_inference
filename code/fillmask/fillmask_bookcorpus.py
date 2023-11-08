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
_logger = logging.getLogger("fillmask_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/fillmask_testing.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models
from huggingface_hub import HfApi, ModelFilter
hf_api= HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="fill-mask", language="en"),
    sort="downloads", direction=-1, limit=8)
fillmask_models = [l.modelId for l in models]

### Load prompting datasets
from datasets import load_dataset, Dataset

def dset_gen():
    dset = load_dataset("bookcorpus", streaming=True, split="train")
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

for model in fillmask_models:
    print(model)
    tracker = EmissionsTracker(project_name=model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/fillmask.csv')
    tracker.start()
    tracker.start_task("load model")
    fillmask = pipeline("fill-mask", model=model, device=0,  trust_remote_code=True )
    model_emissions = tracker.stop_task()
    tokenizer = AutoTokenizer.from_pretrained(model)
    mask = tokenizer.mask_token
    tracker.start_task("query model")
    count = 0
    for d in dset:
        count+=1
        prompt =d['text'].replace('.','')+ mask
        fillmask(prompt)
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
