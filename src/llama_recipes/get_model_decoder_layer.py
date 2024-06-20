from transformers.models.clip.modeling_clip import (
    CLIPVisionEmbeddings,
    CLIPVisionTransformer,
    CLIPEncoderLayer,
    CLIPVisionModel,
)
from transformers.models.idefics2.modeling_idefics2 import (
    Idefics2VisionEmbeddings,
    Idefics2VisionTransformer
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

from megatron_lm.megatron.global_vars import get_args


def get_model_decoder_layer(model_name: str):
    args = get_args()

    if "Llama" in model_name or "Swallow" in model_name:
        return LlamaDecoderLayer
    elif "Mistral" in model_name or "mistral" in model_name:
        return MistralDecoderLayer
    elif "Phi-3" in model_name:
        return Phi3DecoderLayer
    elif "idefics2" in model_name:
        lm_model_type = args.vlm_text_model_type
        if lm_model_type == "llama":
            if args.freeze_vlm_vision_embeddings:
                # avoid `ValueError: Must flatten tensors with uniform `requires_grad` when `use_orig_params=False` error
                return LlamaDecoderLayer, Idefics2VisionTransformer, Idefics2VisionEmbeddings  # type: ignore
            return LlamaDecoderLayer, Idefics2VisionTransformer  # type: ignore

        elif lm_model_type == "mistral":
            if args.freeze_vlm_vision_embeddings:
                return MistralDecoderLayer, Idefics2VisionTransformer, Idefics2VisionEmbeddings
            return MistralDecoderLayer, Idefics2VisionTransformer  # type: ignore
        else:
            raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")

    elif "llava-next" in model_name or "llava-v1.6" in model_name:
        lm_model_type = args.vlm_text_model_type
        if lm_model_type == "llama":
            if args.freeze_vlm_vision_embeddings:
                # avoid `ValueError: Must flatten tensors with uniform `requires_grad` when `use_orig_params=False` error
                return LlamaDecoderLayer, CLIPVisionTransformer, CLIPVisionEmbeddings
            return LlamaDecoderLayer, CLIPVisionTransformer, CLIPEncoderLayer

        elif lm_model_type == "mistral":
            if args.freeze_vlm_vision_embeddings:
                return MistralDecoderLayer, CLIPVisionTransformer, CLIPVisionEmbeddings
            return MistralDecoderLayer, CLIPVisionTransformer, CLIPEncoderLayer

        else:
            raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")

    elif "llava" in model_name or "llava-v1.5" in model_name:
        lm_model_type = args.vlm_text_model_type
        if lm_model_type == "llama":
            if args.freeze_vlm_vision_embeddings:
                return LlamaDecoderLayer, CLIPEncoderLayer, CLIPVisionEmbeddings
            return LlamaDecoderLayer, CLIPEncoderLayer

        elif lm_model_type == "mistral":
            if args.freeze_vlm_vision_embeddings:
                return MistralDecoderLayer, CLIPEncoderLayer, CLIPVisionEmbeddings
            return MistralDecoderLayer, CLIPEncoderLayer

        else:
            raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")

    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
