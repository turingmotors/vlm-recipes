import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as torch_distributed

from transformers.processing_utils import ProcessorMixin
from datasets import load_dataset

from llama_recipes.utils.distributed import print_rank_0
from megatron_lm.megatron.global_vars import get_args, set_sampler


class VisualInstructDataset(Dataset):
    def __init__(
        self,
        processor: ProcessorMixin,  # image_processor & tokenizer
        text_data_path: str,
        image_data_path: str,
        image_token_id: int,
        train: bool = False,
    ) -> None:
        args = get_args()

        self.text_data_path: str = text_data_path
        self.image_data_path: str = image_data_path
        self.max_words: int = args.seq_length
        self.processor = processor
        self.image_token_id = image_token_id

        assert text_data_path == image_data_path, "text and image dataset should be the same"
        self.text_and_image_dataset = load_dataset(
            text_data_path,
            split="train" if train else "test",
        )
        self.text_and_image_dataset = self.text_and_image_dataset.remove_columns(
            ['id', 'words', 'bounding_boxes', 'answer']
        )

    def __len__(self) -> int:
        return len(self.text_and_image_dataset)  # type: ignore

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        example = self.text_and_image_dataset[index]  # type: ignore

        image = example["image"]
        question = example["query"]["en"]  # type: ignore
        answer = random.choice(example["answers"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = self.processor.apply_chat_template(  # type: ignore
            messages,
            add_generation_prompt=False
        )

        # batch (input_ids, attention_mask, pixel_values, pixel_attention_mask)
        batch = self.processor(  # type: ignore
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # delete batch dimension (always batch_size=1)
        for key in batch:
            batch[key] = batch[key].squeeze(0)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX  # type: ignore
        # labels[labels == self.image_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        return batch


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
) -> DataLoader:
    from llama_recipes.utils.sequence_length_warmup import CustomDistributedSampler
    from llama_recipes.utils.checkpoint import load_sampler_state_dict

    args = get_args()

    instruction_dataset = VisualInstructDataset(
        processor=processor,
        text_data_path=text_data_path,
        image_data_path=image_data_path,
        image_token_id=image_token_id,
        train=train,
    )

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
