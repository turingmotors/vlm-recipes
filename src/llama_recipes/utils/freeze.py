import torch


def freeze_vlm_vision_model(model: torch.nn.Module) -> None:
    """Freeze the vision model of the VLM.

    model: Idefics2Model
        vision_model: Idefics2VisionTransformer
            embeddings: Idefics2VisionEmbeddings
            encoder: Idefics2VisionEncoder
                layers: List[Idefics2VisionEncoderLayer]
                    self_attn: Idefics2VisionAttention
                    mlp: Idefics2VisionMLP
                        activation_fn: GELU
                        fc1: Linear
                        fc2: Linear
    """
    vision_model = model.model.vision_model
    for param in vision_model.parameters():
        param.requires_grad = False


def freeze_vlm_vision_embeddings(model: torch.nn.Module) -> None:
    """Freeze the vision embeddings of the VLM.

    model: Idefics2Model
        vision_model: Idefics2VisionTransformer
            embeddings: Idefics2VisionEmbeddings
            encoder: Idefics2VisionEncoder
                layers: List[Idefics2VisionEncoderLayer]
                    self_attn: Idefics2VisionAttention
                    mlp: Idefics2VisionMLP
                        activation_fn: GELU
                        fc1: Linear
                        fc2: Linear
    """
    vision_model = model.model.vision_model
    for param in vision_model.embeddings.parameters():
        param.requires_grad = False


def freeze_vlm_vision_encoder(model: torch.nn.Module) -> None:
    """Freeze the vision encoder of the VLM.

    model: Idefics2Model
        vision_model: Idefics2VisionTransformer
            embeddings: Idefics2VisionEmbeddings
            encoder: Idefics2VisionEncoder
                layers: List[Idefics2VisionEncoderLayer]
                    self_attn: Idefics2VisionAttention
                    mlp: Idefics2VisionMLP
                        activation_fn: GELU
                        fc1: Linear
                        fc2: Linear
    """
    vision_model = model.model.vision_model
    for param in vision_model.encoder.parameters():
        param.requires_grad = False


def freeze_vlm_text_model(model: torch.nn.Module) -> None:
    """Freeze the text model of the VLM.

    model: Idefics2Model
        text_model
            embed_tokens: Embedding
            layers
            norm
            lm_head
    """
    text_model = model.model.text_model
    for param in text_model.parameters():
        param.requires_grad = False
