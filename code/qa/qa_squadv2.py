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
_logger = logging.getLogger("qa_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/qa_squadv2.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load models
hf_api= HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
        language="en",
		task="question-answering"),
    sort="downloads", direction=-1, limit=20)
qa_models = [l.modelId for l in models if 'Bio' not in l.modelId][:8]

def dset_gen():
    dset = load_dataset("squad_v2", split= 'validation', streaming=True)
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

for model in qa_models:
    print(model)
    tracker = EmissionsTracker(project_name=model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/qa_squadv2.csv')
    tracker.start()
    tracker.start_task("load model")
    classifier = pipeline("question-answering", model=model, device=0 )
    model_emissions = tracker.stop_task()
    tracker.start_task("query model")
    count=0
    for sent in dset:
        count+=1
        QA_input = {
        'question': sent['question'],
        'context': sent['context']}
        classifier(QA_input)
    model_emissions = tracker.stop_task()
    print('================'+str(count)+'================')
    _ = tracker.stop()
    torch.cuda.empty_cache()
