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
_logger = logging.getLogger("textgen_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/textgen_oscar_testing.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models
from huggingface_hub import HfApi, ModelFilter
hf_api=HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="text-generation"),
    sort="downloads", direction=-1, limit=15)

text_gen_models = [l.modelId for l in models if 'baichuan' not in l.modelId and 'vicuna' not in l.modelId and 'falcon' not in l.modelId and 'Tranception_Small' not in l.modelId and 'instruct' not in l.modelId][:8]



### Load prompting datasets
from datasets import load_dataset, Dataset

def dset_gen():
    dset = load_dataset("oscar", "unshuffled_deduplicated_en", split= 'train', streaming=True)
    sample = dset.take(1000)
    for row in sample:
      text = ' '.join(row['text'].split()[:20])
      row['text'] = text
      yield row

dataset = Dataset.from_generator(dset_gen)

for tg_model in text_gen_models:
    print(tg_model)
    tracker = EmissionsTracker(project_name=tg_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/textgen_oscar.csv')
    tracker.start()
    tracker.start_task("load model")
    text_gen = pipeline("text-generation", model=tg_model, device=0,  trust_remote_code=True, max_new_tokens=10 )
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count=0
    for d in dataset:
        count+=1
        text_gen(d['text'])
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
