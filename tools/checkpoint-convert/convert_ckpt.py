import argparse
import torch
from transformers import Idefics2ForConditionalGeneration, LlavaNextForConditionalGeneration


def main() -> None:
    """
    Main function to convert a PyTorch checkpoint to a Hugging Face model checkpoint.

    It loads a Hugging Face model based on the specified base model type, updates its state
    with a provided PyTorch checkpoint, and then saves the updated model in Hugging Face format.
    """

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert the PyTorch checkpoint to the Hugging Face checkpoint.")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the Hugging Face base checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the PyTorch checkpoint (e.g., 'model.pth').")
    parser.add_argument("--out", type=str, required=True, help="Output path for the Hugging Face model checkpoint.")

    # Parse the command-line arguments
    args: argparse.Namespace = parser.parse_args()

    # Inform the user that the base Hugging Face model is being loaded
    print(f"Loading Base HF model: {args.base_model}", flush=True)

    # Load the appropriate model based on the base model's name
    if "idefics2" in args.base_model:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif "llava" in args.base_model:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        # Raise an error if the base model is not supported
        raise ValueError(f"{args.base_model} is not supported.")

    # Inform the user that the PyTorch checkpoint is being loaded
    print(f"Loading CKPT: {args.ckpt}", flush=True)
    # Load the state dictionary (model weights) from the specified checkpoint
    state_dict: dict = torch.load(args.ckpt, map_location="cpu")

    # Inform the user that the state dictionary is being loaded into the Hugging Face model
    print("Loading state dict into HF model", flush=True)
    # Load the state dictionary into the Hugging Face model
    model.load_state_dict(state_dict)

    # Inform the user that the Hugging Face model is being saved
    print("Saving HF model", flush=True)
    # Save the updated Hugging Face model to the specified output path
    model.save_pretrained(args.out, safe_serialization=True)


if __name__ == "__main__":
    """
    Example usage:

    BASE_MODEL_CHECKPOINT=/path/to/base-huggingface-checkpoint/idefics2
    CHECK_POINT_PATH=/path/to/pytorch_checkpoint/iter_*******/model.pt
    HF_OUTPUT_PATH=/path/to/huggingface-checkpoint/trained_idefics2_hf

    Command to run:
    python tools/inference/inference.py \
        --base-model $BASE_MODEL_CHECKPOINT \
        --ckpt $CHECK_POINT_PATH \
        --out $HF_OUTPUT_PATH
    """
    main()
