import numpy as np
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


def get_visual_instruction_tuning_dataloader(
    processor: ProcessorMixin,
    text_data_path: str,
    image_data_path: str,
    image_token_id: int,
    train: bool = False,
    dataset_type: str = "TikZ_Instruct",
) -> DataLoader:
    from llama_recipes.utils.checkpoint import load_sampler_state_dict
    from llama_recipes.utils.sequence_length_warmup import CustomDistributedSampler

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
    )
