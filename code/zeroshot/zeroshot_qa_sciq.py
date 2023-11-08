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
_logger = logging.getLogger("zeroshot_qa_sciq")
_channel = logging.FileHandler('/fsx/sashaluc/logs/zeroshot_qa_sciq.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models
def dset_gen():
    dset = load_dataset("sciq", split= 'test', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

qa_df = []
for sent in dset:
    qa_df.append([sent['question'],sent['passage']])

text2text_models = ['google/flan-t5-base', 'google/flan-t5-large','google/flan-t5-xl', 'google/flan-t5-xxl']
text_models= ["bigscience/bloomz-560m", "bigscience/bloomz-1b7", "bigscience/bloomz-7b1", "bigscience/bloomz-3b"]

for m in text2text_models:
    print(m)
    tracker = EmissionsTracker(project_name=m, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/zeroshot_qa_sciq.csv')
    tracker.start()
    tracker.start_task("load model")
    pipe = pipeline("text2text-generation", model=m, device=0)
    model_emissions = tracker.stop_task()
    count=0
    tracker.start_task("query model")
    for p in qa_df:
        pipe(p[0] + '? Find the answer to the question in the following text: "'+p[1]+'". Answer:')
        count+=1
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()

for m in text_models:
    print(m)
    tracker = EmissionsTracker(project_name=m, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/zeroshot_qa_sciq_bloomz.csv')
    tracker.start()
    tracker.start_task("load model")
    pipe = pipeline("text-generation", model=m, device=0)
    model_emissions = tracker.stop_task()
    count=0
    tracker.start_task("query model")
    for p in qa_df:
        pipe(p[0] + '? Find the answer to the question in the following text: "'+p[1]+'". Answer:')
        count+=1
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
