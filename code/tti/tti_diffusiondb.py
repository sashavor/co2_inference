from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer
from diffusers import DiffusionPipeline

from codecarbon import EmissionsTracker
from huggingface_hub import HfApi, ModelFilter
import logging
import torch

# Create a dedicated logger (log name can be the CodeCarbon project name for example)
_logger = logging.getLogger("tti_testing")
_channel = logging.FileHandler('/fsx/sashaluc/logs/tti_testing_diffusiondb.log')
_logger.addHandler(_channel)
_logger.setLevel(logging.INFO)

## Load TTI models
hf_api = HfApi()
models = hf_api.list_models(
    filter=ModelFilter(
		task="text-to-image"),
    sort="downloads", direction=-1, limit=13)
tti_models = [l.modelId for l in models if l.modelId != 'stabilityai/stable-diffusion-xl-refiner-1.0' and l.modelId != 'stable-diffusion-inpainting'][:6]
tti_models.append("nota-ai/bk-sdm-tiny")
tti_models.append("segmind/tiny-sd")

### Load prompting datasets

def dset_gen():
    dset = load_dataset("poloclub/diffusiondb", "2m_first_1k", streaming=True, split="train")
    sample = dset.take(1000)
    for row in sample:
        yield row

dset = Dataset.from_generator(dset_gen)

for tti_model in tti_models:
    print(tti_model)
    tracker = EmissionsTracker(project_name=tti_model, measure_power_secs=1, logging_logger=_logger, output_file='/fsx/sashaluc/emissions/tti_diffusiondb.csv')
    tracker.start()
    tracker.start_task("load model")
    pipe = DiffusionPipeline.from_pretrained(tti_model )
    model_emissions = tracker.stop_task()
    print('loaded model')
    tracker.start_task("transfer model to GPU")
    pipe = pipe.to("cuda")
    model_emissions = tracker.stop_task()
    shorter_prompts=[]
    for d in dset:
        prompt = ' '.join(d['prompt'].split()[:50])
        shorter_prompts.append(prompt)
    tracker.start_task("query model")
    count=0
    for prompt in shorter_prompts:
            pipe(prompt)
            count+=1
    print('================'+str(count)+'================')
    model_emissions = tracker.stop_task()
    _ = tracker.stop()
    torch.cuda.empty_cache()
