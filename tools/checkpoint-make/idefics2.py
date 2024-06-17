from transformers import (
    AutoTokenizer,
    CLIPModel,
    Idefics2Config,
    Idefics2Model,
    LlamaForCausalLM,
    MistralForCausalLM,
    SiglipModel
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="idefics2", help="model name")
    parser.add_argument("--vision-encoder-path", type=str, default="", help="vision encoder path")
    parser.add_argument("--language-model-path", type=str, default="", help="language model path")
    parser.add_argument("--save-path", type=str, default="", help="save path")
    parser.add_argument("--use-pretrained-connector", action="store_true", help="use pretrained connector")
    parser.add_argument("--pre-trained-ckpt-path", type=str, default="", help="pretrained checkpoint path")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # vision encoder
    if "siglip" in args.vision_encoder_path:
        vision_model = SiglipModel.from_pretrained(
            pretrained_model_name_or_path=args.vision_encoder_path
        )
    elif "clip" in args.vision_encoder_path:
        vision_model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=args.vision_encoder_path
        )
    else:
        raise ValueError(f"Invalid vision model: {args.vision_encoder_path}")

    # language model
    if "CodeLlama" in args.language_model_path:
        language_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.language_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    elif "Codestral" in args.language_model_path:
        language_model = MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.language_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    elif "deepseek-coder" in args.language_model_path:
        language_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.language_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(args.language_model_path)
    else:
        raise ValueError(f"Invalid language model: {args.language_model_path}")


    # tokenizer vocab extension
    # https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
    special_tokens = {
        "additional_special_tokens": [
            "<fake_token_around_image>",
            "<image>",
            "<end_of_utterance>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict=special_tokens)  # type: ignore
    language_model.resize_token_embeddings(len(tokenizer))  # type: ignore

    # model load
    config = Idefics2Config(
        use_cache=False,
        tie_word_embeddings=False,
    )

    model = Idefics2Model(
        config=config,
    )

    # change vision encoder
    if "siglip" in args.vision_encoder_path or "clip" in args.vision_encoder_path:
        for name, _ in model.vision_model.named_parameters():
            if "model.vision_model.encoder" in name:
                layer_idx = int(name.split(".")[4])
                param = vision_model.vision_model.encoder.layers[layer_idx].state_dict()  # type: ignore
                model.vision_model.encoder.layers[layer_idx].load_state_dict(param)
    # vision position embeddings

    # change language model
    if "CodeLlama" in args.language_model_path or "deepseek-coder" in args.language_model_path:
        model.text_model.load_state_dict(language_model.state_dict())  # type: ignore
    elif "Codestral" in args.language_model_path:
        model.text_model.load_state_dict(language_model.state_dict())  # type: ignore

    # connector
    if args.use_pretrained_connector:
        pre_trained_model = Idefics2Model.from_pretrained(
            pretrained_model_name_or_path=args.pre_trained_ckpt_path
        )
        model.connector.load_state_dict(
            pre_trained_model.connector.state_dict()  # type: ignore
        )

    # save model
    model.save_pretrained(args.save_path, safe_serialization=True)


if __name__ == "__main__":
    main()
