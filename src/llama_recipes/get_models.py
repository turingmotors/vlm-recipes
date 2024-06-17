from transformers import (
    Idefics2Model,
    Idefics2ForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaForConditionalGeneration,
)
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2Config,
    Idefics2VisionConfig,
    Idefics2PerceiverConfig,
)

from llama_recipes.utils.distributed import is_rank_0
import torch
from megatron_lm.megatron.global_vars import get_args


def get_model(
    model_name: str, use_cache: bool = False
) -> Idefics2ForConditionalGeneration | LlavaNextForConditionalGeneration | LlavaForConditionalGeneration:
    args = get_args()

    if "idefics2" in model_name:
        # Idefics2
        #  vision model: SIGLIP
        #  text model: Mistral
        # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py#L1464
        if args.vlm_vision_model_type == "siglip":
            args.vlm_vision_config = "idefics2"

        model = Idefics2ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            use_cache=use_cache,
            tie_word_embeddings=False,
            perceiver_config={
                "model_type": args.vlm_perceiver_model_type,
                "hidden_act": args.vlm_perceiver_hidden_act,
                "resampler_n_latents": args.vlm_perceiver_resampler_n_latents,
                "resampler_depth": args.vlm_perceiver_resampler_depth,
                "resampler_n_heads": args.vlm_perceiver_resampler_n_heads,
                "resampler_head_dim": args.vlm_perceiver_resampler_head_dim,
                "num_key_value_heads": args.vlm_perceiver_num_key_value_heads,
                "attention_dropout": args.vlm_perceiver_attention_dropout,
            },
            text_config={
                "max_position_embeddings": args.seq_length,
                "sliding_window": args.seq_length,
                "model_type": args.vlm_text_model_type,
                "pad_token_id": args.pad_token_id,
                "rms_norm_eps": args.rms_norm_eps,
                "vocab_size": args.vocab_size,
                "use_cache": use_cache,
                "hidden_size": args.vlm_text_hidden_size,
                "intermediate_size": args.vlm_text_intermediate_size,
                "num_attention_heads": args.vlm_text_num_attention_heads,
                "num_hidden_layers": args.vlm_text_num_hidden_layers,
                "num_key_value_heads": args.vlm_text_num_key_value_heads,
                "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
                "rope_theta": args.vlm_text_rope_theta,
            },
            vision_config={
                "hidden_size": args.vlm_vision_hidden_size,
                "image_size": args.vlm_vision_image_size,
                "intermediate_size": args.vlm_vision_intermediate_size,
                "model_type": args.vlm_vision_model_type,
                "num_attention_heads": args.vlm_vision_num_attention_heads,
                "num_hidden_layers": args.vlm_vision_num_hidden_layers,
                "patch_size": args.vlm_vision_patch_size,
            },
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

        return model  # type: ignore

    elif "llava-next" in model_name or "llava-v1.6" in model_name:
        # ref https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/blob/main/config.json
        # transformers: https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/llava_next/modeling_llava_next.py#L345
        model = LlavaNextForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            vocab_size=args.vocab_size,
            text_config={
                "max_position_embeddings": args.seq_length,
                "sliding_window": args.sliding_window_size,
                "model_type": args.vlm_text_model_type,
                "vocab_size": args.vocab_size,
                "use_cache": use_cache,
                "hidden_size": args.vlm_text_hidden_size,
                "intermediate_size": args.vlm_text_intermediate_size,
                "num_attention_heads": args.vlm_text_num_attention_heads,
                "num_hidden_layers": args.vlm_text_num_hidden_layers,
                "num_key_value_heads": args.vlm_text_num_key_value_heads,
                "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
                "rope_theta": args.vlm_text_rope_theta,
            },
            vision_config={
                "hidden_size": args.vlm_vision_hidden_size,
                "image_size": args.vlm_vision_image_size,
                "intermediate_size": args.vlm_vision_intermediate_size,
                "model_type": args.vlm_vision_model_type,
                "num_attention_heads": args.vlm_vision_num_attention_heads,
                "num_hidden_layers": args.vlm_vision_num_hidden_layers,
                "patch_size": args.vlm_vision_patch_size,
                "projection_dim": args.vlm_vision_projection_dim,
                "vocab_size": args.vlm_vision_vocab_size,
            },
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

        return model  # type: ignore

    elif "llava" in model_name:
        # ref: https://huggingface.co/liuhaotian/llava-v1.5-13b/blob/main/config.json
        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            tie_word_embeddings=False,
            text_config={
                "max_position_embeddings": args.seq_length,
                "model_type": args.vlm_text_model_type,
                "vocab_size": args.vocab_size,
                "use_cache": use_cache,
                "hidden_size": args.vlm_text_hidden_size,
                "intermediate_size": args.vlm_text_intermediate_size,
                "num_attention_heads": args.vlm_text_num_attention_heads,
                "num_hidden_layers": args.vlm_text_num_hidden_layers,
                "num_key_value_heads": args.vlm_text_num_key_value_heads,
                "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
                "rope_theta": args.vlm_text_rope_theta,
            },
            vision_config={
                "hidden_size": args.vlm_vision_hidden_size,
                "image_size": args.vlm_vision_image_size,
                "intermediate_size": args.vlm_vision_intermediate_size,
                "model_type": args.vlm_vision_model_type,
                "num_attention_heads": args.vlm_vision_num_attention_heads,
                "num_hidden_layers": args.vlm_vision_num_hidden_layers,
                "patch_size": args.vlm_vision_patch_size,
                "projection_dim": args.vlm_vision_projection_dim,
                "vocab_size": args.vlm_vision_vocab_size,
            },
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

        return model  # type: ignore

    else:
        raise NotImplementedError("model not implemented")
