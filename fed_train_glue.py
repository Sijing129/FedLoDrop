import os
os.environ["WANDB_MODE"] = "offline"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
import argparse
import warnings
import os
from datetime import datetime
import numpy as np
import wandb
from train_eval import *
from fed_agg import *
import json
from utils import *
import random

#dropout
def random_index(shape_scale: int, portion: float = 0.0):
    _mask_index = random.sample([index for index in range(shape_scale)], int(portion * shape_scale))
    return sorted(_mask_index)


def lora_dropout(lora_layer_list: [torch.Tensor], dropout_rate: float = 0.0, dim: int = 0) -> None:
    """
    将name中有lora的参数矩阵全部放在一个列表中处理
    :param lora_layer_list: 所有要drop的lora层
    :param dropout_rate: drop率
    :param dim: 要drop行,或者列(对应是drop loraA还是loraB)
    :return:
    """
    for lora_layer in lora_layer_list:
        mask_index = random_index(lora_layer.shape[dim], portion=dropout_rate)

        if dim == 0:
            lora_layer[mask_index] = 0.0
        elif dim == 1:
            lora_layer[:, mask_index] = 0.0


def drop_peft_model(model, dropout_rate: float = 0.0, lora_type: str = "A"):
    if lora_type == "A":
        # drop by columns
        dim = 1
        lora_name = "lora_A"
    elif lora_type == "B":
        # drop by rows
        dim = 0
        lora_name = "lora_B"
    else:
        raise ValueError("Lora type must be \"A\" or \"B\"")

    layer_list = []
    for name, param in model.named_parameters():
        if lora_name in name:
            layer_list.append(param.data)

    lora_dropout(layer_list, dropout_rate=dropout_rate, dim=dim)


parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--task", type=str, default="cola", help="GLUE task to fine-tune on"
)
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")  #本来64
parser.add_argument(
    "--agg_type", type=str, default="ours", help="Type of aggregation"
)
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Maximum sequence length"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")  #本来42

args = parser.parse_args()

wandb.init(project="project_name", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def federated_learning(task):

    train_data, val_data, test_data = load_and_preprocess_data(task)
    # print('111111111111111111111111111111111')
    # print(len(train_data))

    print(type(train_data["labels"]))
    num_labels = len(set(train_data["labels"]))

    if args.task == "stsb":
        num_labels = 1

    client_dataloaders = create_client_dataloaders(train_data, args)
    val_dataloader = create_dataloader(val_data, args)

    max_metric_1 = 0
    max_metric_2 = 0

    if args.agg_type == "ffa":
        global_model = create_peft_FFA_model(num_labels, args)
    else:
        global_model = create_peft_model(num_labels, args)

    client_models = []

    for i in range(args.num_clients):

        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(num_labels, args)
        else:
            client_model = create_peft_model(num_labels, args)
            # drop_peft_model(client_model, dropout_rate=0.9, lora_type="A")    #加入drop
            # drop_peft_model(client_model, dropout_rate=0.9, lora_type="B")

        client_models.append(client_model)

    for round in range(args.rounds):
        print(f"Round {round + 1}/{args.rounds}")

        client_model_state_dicts = []

        preserving_rate = [0.7, 0.7, 0.7]   # 每个客户端的保留率，可以根据需要进行调整
        dropout_rate = np.ones(args.num_clients)- preserving_rate

        for i in range(args.num_clients):
            client_model = client_models[i]
            client_model.load_state_dict(global_model.state_dict())
            client_model_state_dict = train_client(
                client_model, client_dataloaders[i], dropout_rate[i], args
            )
            client_model_state_dicts.append(client_model_state_dict)

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ours":
            global_model = aggregate_models_ours(global_model, client_models, args)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        max_metric_1, max_metric_2 = evaluate_global_model(
            global_model, val_dataloader, args, max_metric_1, max_metric_2
        )


# Main execution
if __name__ == "__main__":
    task = args.task
    model = federated_learning(task)