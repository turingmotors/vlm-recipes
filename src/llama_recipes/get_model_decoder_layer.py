from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer

from megatron_lm.megatron.global_vars import get_args


def get_model_decoder_layer(
    model_name: str,
) -> type[LlamaDecoderLayer] | type[MistralDecoderLayer] | type[Phi3DecoderLayer]:
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
            return LlamaDecoderLayer, Idefics2VisionTransformer  # type: ignore
        elif lm_model_type == "mistral":
            return MistralDecoderLayer, Idefics2VisionTransformer  # type: ignore
        else:
            raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
