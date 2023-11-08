from transformers import pipeline
from datasets import Dataset, load_dataset
from transformers import pipeline, AutoTokenizer

from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, ModelFilter
import logging
import torch
import huggingface_hub
import einops

import os


# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("zeroshot_sentiment_tomatoes")
_channel = logging.FileHandler('/fsx/sashaluc/logs/zeroshot_sentiment_tomatoes.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models

def dset_gen():
    dset = load_dataset("rotten_tomatoes", split= 'validation', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

prompts=[]
for d in dset:
    text = d['text']
    prompts.append(text)

text2text_models = ['google/flan-t5-base', 'google/flan-t5-large','google/flan-t5-xl', 'google/flan-t5-xxl']
text_models= ["bigscience/bloomz-560m", "bigscience/bloomz-1b7", "bigscience/bloomz-7b1", "bigscience/bloomz-3b"]

for m in text2text_models:
    print(m)
    tracker = EmissionsTracker(project_name=m, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/zeroshot_sentiment_tomatoes_bloomz.csv')
    tracker.start()
    tracker.start_task("load model")
    pipe = pipeline("text2text-generation", model=m, device=0 )
    model_emissions = tracker.stop_task()
    count=0
    tracker.start_task("query model")
    for p in prompts:
        pipe('what is the binary sentiment of the following sentence? '+p)
        count+=1
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()


for m in text_models:
    print(m)
    tracker = EmissionsTracker(project_name=m, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/zeroshot_sentiment_tomatoes_bloomz.csv')
    tracker.start()
    tracker.start_task("load model")
    pipe = pipeline("text-generation", model=m, device=0 , max_new_tokens = 1, trust_remote_code=True)
    model_emissions = tracker.stop_task()
    count=0
    tracker.start_task("query model")
    for p in prompts:
        pipe('what is the binary sentiment of the following sentence? '+p)
        count+=1
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
