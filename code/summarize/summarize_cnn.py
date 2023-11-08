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
_logger = logging.getLogger("summarize_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/summarize_testing_cnn_final.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models
hf_api= HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="summarization"),
    sort="downloads", direction=-1, limit=8)
summarization_models = [l.modelId for l in models]

### Load prompting datasets
from datasets import load_dataset, Dataset

def dset_gen():
    dset = load_dataset("cnn_dailymail", "3.0.0", split= 'test', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dataset = Dataset.from_generator(dset_gen)

for sum_model in summarization_models:
    print(sum_model)
    tracker = EmissionsTracker(project_name=sum_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/summarize_cnn_15.csv')
    tracker.start()
    tracker.start_task("load model")
    summarize = pipeline("summarization", model=sum_model, device=0,  trust_remote_code=True)
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count=0
    for d in dset:
        count+=1
        summarize(d['article'], max_length= 15, min_length=10)
    print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()
