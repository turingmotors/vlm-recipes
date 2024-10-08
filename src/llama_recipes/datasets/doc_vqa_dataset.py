import random

import torch
from torch.utils.data import Dataset

from transformers.processing_utils import ProcessorMixin
from datasets import load_dataset

from llama_recipes.utils.distributed import print_rank_0
from megatron_lm.megatron.global_vars import get_args


# ref: https://huggingface.co/datasets/nielsr/docvqa_1200_examples
class DocumentVQADataset(Dataset):
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
        self.max_seq_length: int = args.seq_length
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

        if self.processor.__class__.__name__ == "Idefics2Processor":
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
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            messages = [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

        if self.processor.__class__.__name__ == "Idefics2Processor":
            # ref: https://huggingface.co/HuggingFaceM4/idefics2-8b/blob/main/processor_config.json#L2
            text = self.processor.apply_chat_template(  # type: ignore
                messages,
                add_generation_prompt=False
            )
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            # ref: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/main/preprocessor_config.json
            # ref: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/main/tokenizer_config.json#L48
            text = self.processor.tokenizer.apply_chat_template(  # type: ignore
                messages,
                add_generation_prompt=False,
                tokenize=False  # 指定しないと自動でtokenizeされてしまう
            )

        # batch (input_ids, attention_mask, pixel_values, pixel_attention_mask)
        if self.processor.__class__.__name__ == "Idefics2Processor":
            batch = self.processor(  # type: ignore
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            batch = self.processor(  # type: ignore
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )

        # delete batch dimension (always batch_size=1)
        for key in batch:
            batch[key] = batch[key].squeeze(0)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX  # type: ignore
        # labels[labels == self.image_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        return batch
