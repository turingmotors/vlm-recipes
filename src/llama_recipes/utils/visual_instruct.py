import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
import torch.distributed as torch_distributed
from torch.utils.data import DataLoader
from transformers.processing_utils import ProcessorMixin

from llama_recipes.utils.distributed import print_rank_0
from megatron_lm.megatron.global_vars import get_args, set_sampler


def worker_init_fn(worker_id: int) -> None:
    import random

    args = get_args()

    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pad_tensor(tensor, target_size):
    # print(f"DEBUG: tensor.size()={tensor.size()}, target_size={target_size}", flush=True)

    if tensor.dim() == 4:
        _, _, h, w = tensor.size()
    elif tensor.dim() == 3:
        _, h, w = tensor.size()
    else:
        raise ValueError("Unsupported tensor dimension: {}".format(tensor.dim()))

    padding = [0, target_size[1] - w, 0, target_size[0] - h]  # padding right, bottom
    return F.pad(tensor, padding, "constant", 0)


def custom_collate_fn(batch):
    # resizes all tensors to the same size
    collated_batch = {}
    for key in batch[0].keys():
        if key == "pixel_values" or key == "pixel_attention_mask":
            # get the max height and width (in the batch)
            # batch.shape = (batch_size, rgb, height, width) or (batch_size, height, width)
            # item = (1, 3, height, width) or (1, height, width)
            if batch[0][key].dim() == 3:
                max_height = max(item[key].size(-2) for item in batch)
                max_width = max(item[key].size(-1) for item in batch)
            elif batch[0][key].dim() == 4:
                max_height = max(item[key].size(-2) for item in batch)
                max_width = max(item[key].size(-1) for item in batch)
            else:
                raise ValueError(f"Invalid dim: {batch[0][key].dim()}")

            target_size = (max_height, max_width)

            # pad all tensors to the same size
            padded_tensors = [
                pad_tensor(item[key], target_size) for item in batch
            ]
            collated_batch[key] = torch.stack(padded_tensors)
        else:
            # resize all tensors to the same size
            max_size = max(item[key].size(0) for item in batch)
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                padding_size = (0, max_size - tensor.size(0))
                padded_tensor = torch.nn.functional.pad(tensor, padding_size)
                padded_tensors.append(padded_tensor)
            collated_batch[key] = torch.stack(padded_tensors)

        # stack all padded tensors
        collated_batch[key] = torch.stack(padded_tensors)

    return collated_batch


def get_visual_instruction_tuning_dataloader(
    processor: ProcessorMixin,
    text_data_path: str,
    image_data_path: str,
    image_token_id: int,
    train: bool = False,
    dataset_type: str = "TikZ_Instruct",
) -> DataLoader:
    from llama_recipes.utils.checkpoint import load_sampler_state_dict
    from llama_recipes.utils.sequence_length_warmup import \
        CustomDistributedSampler

    args = get_args()

    if dataset_type == "TikZ_Instruct":
        from llama_recipes.datasets.tikz_instruct import TikZInstructDataset
        instruction_dataset = TikZInstructDataset(
            processor=processor,
            text_data_path=text_data_path,
            image_data_path=image_data_path,
            image_token_id=image_token_id,
            train=train,
        )
    elif dataset_type == "LLaVA_PreTrain":
        from llama_recipes.datasets.llava_pretrain import LLaVAPraTrainDataset
        instruction_dataset = LLaVAPraTrainDataset(
            processor=processor,
            text_data_path=text_data_path,
            image_data_path=image_data_path,
            image_token_id=image_token_id,
            train=train,
        )
    elif dataset_type == "DocumentVQA":
        from llama_recipes.datasets.doc_vqa_dataset import DocumentVQADataset
        instruction_dataset = DocumentVQADataset(
            processor=processor,
            text_data_path=text_data_path,
            image_data_path=image_data_path,
            image_token_id=image_token_id,
            train=train,
        )
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    if train:
        args.instruction_dataset_size = len(instruction_dataset)
        print_rank_0(f"Visual Instruction dataset size: {args.instruction_dataset_size}")

    train_sampler = CustomDistributedSampler(
        dataset=instruction_dataset,
        rank=torch_distributed.get_rank(),
        num_replicas=torch_distributed.get_world_size(),
        shuffle=True,
        seed=args.seed,
    )

    if args.load:
        load_sampler_state_dict(sampler=train_sampler, path=args.load)

    set_sampler(sampler=train_sampler)

    return DataLoader(
        instruction_dataset,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=custom_collate_fn,
    )
