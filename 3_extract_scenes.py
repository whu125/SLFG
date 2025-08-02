import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config(config_path='configs/paths.json'):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_llm(model_name, device_map):
    """Loads and sets up the language model."""
    print(f"Loading LLM from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def extract_scenes_from_caption(data, model, tokenizer, device):
    """
    Uses an LLM to break down a long caption into distinct scenes.
    """
    caption = data["caption"]
    
    extraction_prompt_template = """
Given the caption of a video, extract the individual scenes described in it. 
Each scene should consist of a few sentences that include the key elements of that scene.

Caption: {caption}

Extracted Scenes:
"""
    prompt = extraction_prompt_template.format(caption=caption)

    messages = [
        {"role": "system", "content": "You are a helpful assistant trained to break down long descriptions into smaller, meaningful segments."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
        
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    # Split response into scenes
    scenes = [scene.strip() for scene in response.split("\n") if scene.strip()]

    return {
        "video": data["video"],
        "tag": data["tag"],
        "start_time": data["start_time"],
        "end_time": data["end_time"],
        "scenes": scenes
    }

def main(args):
    config = load_config()
    output_dir = config['output_dir']
    model_checkpoints_dir = config['model_checkpoints_dir']

    captions_file = os.path.join(output_dir, "captions.json")
    output_file = os.path.join(output_dir, "scenes.json")
    model_name = os.path.join(model_checkpoints_dir, "Qwen2.5-7B-Instruct")
    
    device = f"cuda:{args.gpu_id}"
    device_map = {"": device}

    model, tokenizer = setup_llm(model_name, device_map)

    with open(captions_file, "r", encoding='utf-8') as f:
        caption_dataset = json.load(f)

    all_scenes_data = []
    for data in tqdm(caption_dataset, desc="Extracting Scenes"):
        try:
            scene_data = extract_scenes_from_caption(data, model, tokenizer, device)
            all_scenes_data.append(scene_data)
        except Exception as e:
            print(f"Error processing video {data.get('video', 'UNKNOWN')}: {e}")
            continue

    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump(all_scenes_data, outfile, indent=4, ensure_ascii=False)

    print(f"Scene extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual scenes from video captions using an LLM.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use.")
    args = parser.parse_args()
    main(args)