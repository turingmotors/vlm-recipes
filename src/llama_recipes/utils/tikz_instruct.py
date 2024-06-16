from os import truncate
import PIL

import PIL.PngImagePlugin
import torch
from torch.utils.data import Dataset

from transformers.processing_utils import ProcessorMixin
from datasets import load_dataset

from llama_recipes.utils.distributed import print_rank_0
from megatron_lm.megatron.global_vars import get_args


class TikZInstructDataset(Dataset):
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
        self.text_and_image_dataset = self.text_and_image_dataset.remove_columns(  # dict_keys(['caption', 'code', 'image', 'pdf', 'uri', 'origin', 'date'])
            ['uri', 'origin', 'date', 'pdf']
        )

    def __len__(self) -> int:
        return len(self.text_and_image_dataset)  # type: ignore

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        example = self.text_and_image_dataset[index]  # type: ignore

        caption: str = example["caption"]  # type: ignore
        code: str = example["code"]  # type: ignore
        image: PIL.PngImagePlugin.PngImageFile = example["image"]  # type: ignore

        if self.processor.__class__.__name__ == "Idefics2Processor":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a TikZ image and caption of a TikZ image."},
                        {"type": "text", "text": caption},
                        {"type": "image"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": code}
                    ]
                }
            ]
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            messages = [
                {
                    "role": "user",
                    "content": "Here is a TikZ image and caption of a TikZ image.\n\n" + caption
                },
                {
                    "role": "assistant",
                    "content": code
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
                tokenize=False
            )

        # batch (input_ids, attention_mask, pixel_values, pixel_attention_mask)
        # ImageInput: PIL.Image.Image | np.ndarray | torch.Tensor | List[PIL.Image.Image] | List[np.ndarray] | List[torch.Tensor] (この型を満たすもの)
        if self.processor.__class__.__name__ == "Idefics2Processor":
            batch = self.processor(  # type: ignore
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_words,
            )
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            batch = self.processor(  # type: ignore
                text=text,
                images=image,
                return_tensors="pt",
            )

        # delete batch dimension (always batch_size=1)
        for key in batch:
            batch[key] = batch[key].squeeze(0)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX  # type: ignore
        # labels[labels == self.image_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        return batch
