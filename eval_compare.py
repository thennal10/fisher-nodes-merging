import os
import json
import torch
from copy import deepcopy
from datetime import datetime
from calc_fisher import calculate_fisher
from itertools import combinations
from fisher_nodes import FisherNodeWrapper
from model_evaluator import ModelEvaluator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# get config from config.json
with open("config.json") as f:
    config = json.load(f)


all_tasks = config["checkpoint_names"].keys()
all_checkpoints = config["checkpoint_names"].values()

task_vectors = {}

tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], max_len=512)
evaluator = ModelEvaluator(device=config["device"])

if config["use_mask"] and not config["use_saved"]:
    print("Getting Fisher information...")
    for task, checkpoint_name in config["checkpoint_names"].items():
        checkpoint = AutoModelForSequenceClassification.from_pretrained(checkpoint_name)
        print(f"Getting Fisher information for {task}...")
        neuron_mask, head_mask = calculate_fisher(
            model=deepcopy(checkpoint),
            task_name=task,
            tokenizer=tokenizer,
            num_samples=config["num_samples"], 
            device=config["device"], 
            seed=config["seed"])

        save_dir = os.path.join(
            "saved_fisher",
            task,
            checkpoint_name,
            str(config["num_samples"]), 
            str(config["seed"])
            )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(neuron_mask, os.path.join(save_dir, "neuron_mask.pt"))
        torch.save(head_mask, os.path.join(save_dir, "head_mask.pt"))

metrics = config
metrics['metrics'] = {}

print("Evaluating...")
for tasks in combinations(all_tasks, config["num_at_once"]):
    print(f"Evaluating {tasks}")
    metrics['metrics']["_".join(tasks)] = {}

    task_vectors = []
    for task in tasks:
        checkpoint_name = config["checkpoint_names"][task]
        checkpoint = AutoModelForSequenceClassification.from_pretrained(checkpoint_name)
        if config["use_mask"]:
            save_dir = os.path.join(
                "saved_fisher",
                task,
                checkpoint_name,
                str(config["num_samples"]), 
                str(config["seed"])
                )
            neuron_mask = torch.load(os.path.join(save_dir, "neuron_mask.pt"))
            head_mask = torch.load(os.path.join(save_dir, "head_mask.pt"))
        else:
            neuron_mask = torch.ones(checkpoint.base_model.config.num_hidden_layers, checkpoint.base_model.config.intermediate_size)
            head_mask = torch.ones(checkpoint.base_model.config.num_hidden_layers, checkpoint.base_model.config.num_attention_heads)
        vec = FisherNodeWrapper(finetuned=checkpoint, neuron_mask=neuron_mask, head_mask=head_mask, use_universal=config["use_universal"])
        task_vectors.append(vec)
    
    if len(tasks) == 1:
        vec = deepcopy(task_vectors[0])
    else:
        vec = sum(task_vectors)
    for task in tasks:
        # use the task model with head weights intact
        task_model = AutoModelForSequenceClassification.from_pretrained(config["checkpoint_names"][task])
        new_model = vec.apply_to(task_model)
        new_model = new_model.to(config["device"])
        metric = evaluator.evaluate(new_model, tokenizer, task, batch_size=8)
        metrics['metrics']["_".join(tasks)][task] = metric
        print(f"{task}: {metric}")

# Save the results as json
os.makedirs("metrics", exist_ok=True)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
metric_filename = "_".join([current_time, config["run_name"]]) + ".json"
with open(os.path.join("metrics", metric_filename), "w") as f:
    json.dump(metrics, f, indent=4)