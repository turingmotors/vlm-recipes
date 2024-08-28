import argparse
import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.image_utils import load_image


def main() -> None:
    """
    Main function to load a model and processor, process the image and prompt, and generate text based on the image and prompt.
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate inference results based on an image and a prompt.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained vlm.")
    parser.add_argument("--processor-path", type=str, required=True, help="Path to the corresponding processor.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the task.")
    # Parse the arguments
    args = parser.parse_args()

    # Load model based on the provided model path
    if "idefics2" in args.model_path.lower():
        model = Idefics2ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    elif "llava" in args.model_path.lower():
        model = LlavaNextForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"{args.model_path} is not supported.")

    # Load the processor for the corresponding model
    processor = AutoProcessor.from_pretrained(args.processor_path)

    # Load and preprocess the image
    image1 = load_image(args.image_path)

    # Construct the input messages with the image and the text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ]
        },
    ]

    # Apply chat template to format the inputs
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=False
    )

    # Tokenize text and process the image for the model
    inputs = processor(text=text, images=image1, return_tensors="pt")
    inputs = {k: v.to(device=model.device) for k, v in inputs.items()}

    # Set the end-of-sequence (EOS) tokens
    eos_token_id = processor.tokenizer.eos_token_id
    end_of_utterance_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    eos_token_id_list = [eos_token_id, end_of_utterance_token_id]

    # Generate response from the model using sampling techniques
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.99,
        top_p=0.95,
        do_sample=True,
        eos_token_id=eos_token_id_list
    )

    # Decode and print the generated response
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f'Generated texts:\n{generated_texts}')


if __name__ == "__main__":
    """
    Example usage:

    HF_MODEL_PATH=/path/to/huggingface-checkpoint/idefics2
    HF_PROCESSOR_PATH=/path/to/huggingface-processor/idefics2
    IMAGE_PATH=images/drive_situation_image.jpg
    PROMPT="In the situation in the image, is it permissible to start the car when the light turns green?"

    Command to run:
    python tools/inference/inference.py \
        --model-path $HF_MODEL_PATH \
        --processor-path $HF_PROCESSOR_PATH \
        --image-path $IMAGE_PATH \
        --prompt $PROMPT
    """
    main()
