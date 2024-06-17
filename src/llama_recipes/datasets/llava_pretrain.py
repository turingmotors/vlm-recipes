import json
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers.processing_utils import ProcessorMixin

from llama_recipes.utils.distributed import print_rank_0
from megatron_lm.megatron.global_vars import get_args


# ref: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py#L692
class LLaVAPraTrainDataset(Dataset):
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

        self.text_dataset = json.load(open(text_data_path, "r"))
        # id, image(path: str), conversations: (from, value)

    def __len__(self) -> int:
        return len(self.text_dataset)  # type: ignore

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        example = self.text_dataset[index]  # type: ignore

        conversations: dict = example["conversations"]
        image_path: str = example["image"]
        image_path = self.image_data_path + "/" + image_path

        image = Image.open(image_path)

        if self.processor.__class__.__name__ == "Idefics2Processor":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": conversations[0]["value"]},  # \n<image> が文末にある
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": conversations[1]["value"]},
                    ]
                }
            ]
        elif self.processor.__class__.__name__ == "LlavaNextProcessor":
            # <image> となっているところに画像の情報が入る
            messages = [
                {
                    "role": "user",
                    "content": conversations[0]["value"]  # \n<image> が文末にある
                },
                {
                    "role": "assistant",
                    "content": conversations[1]["value"]
                }
            ]
        elif self.processor.__class__.__name__ == "LlavaProcessor":
            messages = ""
            messages += conversations[0]["value"] + "\n"
            messages += conversations[1]["value"]
        else:
            raise ValueError(f"Invalid processor: {self.processor.__class__.__name__}")

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
        elif self.processor.__class__.__name__ == "LlavaProcessor":
            # no chat template
            text = messages
        else:
            raise ValueError(f"Invalid processor: {self.processor.__class__.__name__}")

        # batch (input_ids, attention_mask, pixel_values, pixel_attention_mask)
        # ImageInput: PIL.Image.Image | np.ndarray | torch.Tensor | List[PIL.Image.Image] | List[np.ndarray] | List[torch.Tensor] (この型を満たすもの)
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
        elif self.processor.__class__.__name__ == "LlavaProcessor":
            batch = self.processor(  # type: ignore
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
        else:
            raise ValueError(f"Invalid processor: {self.processor.__class__.__name__}")

        # delete batch dimension (always batch_size=1)
        for key in batch:
            batch[key] = batch[key].squeeze(0)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX  # type: ignore
        # labels[labels == self.image_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        # print(f"DEBUG: input_ids.shape: {batch['input_ids'].shape}, attention_mask.shape: {batch['attention_mask'].shape}, pixel_values.shape: {batch['pixel_values'].shape}, pixel_attention_mask.shape: {batch['pixel_attention_mask'].shape}, labels.shape: {batch['labels'].shape}", flush=True)

        return batch