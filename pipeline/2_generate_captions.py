import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import warnings

warnings.filterwarnings("ignore")

def load_config(config_path='configs/paths.json'):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_model(model_path, device_map):
    """Loads and sets up the LLaVA model."""
    print(f"Loading model from: {model_path}")
    model_name = get_model_name_from_path(model_path)
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=device_map, attn_implementation="sdpa"
        )
        model.eval()
        return tokenizer, model, image_processor
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def generate_caption_for_segment(segment_info, video_path, model, tokenizer, image_processor, device, frames_per_segment=16):
    """
    Generates a detailed text caption for a single video segment.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_video_frames = len(vr)

        start_frame = int(segment_info['start_time'] * fps)
        end_frame = min(int(segment_info['end_time'] * fps), total_video_frames - 1)
        
        # Uniformly sample frames from the segment
        frame_indices = np.linspace(start_frame, end_frame, frames_per_segment, dtype=int)
        frame_indices = [idx for idx in frame_indices if idx < total_video_frames]

        if not frame_indices:
            return None

        frames = vr.get_batch(frame_indices).asnumpy()
        image_tensors = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(device)

        # Build the prompt
        question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe all the scenes you have seen in as much detail as possible."
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # Generate caption
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=False,
                max_new_tokens=2048,
                modalities=["video"],
            )
        caption = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        return {
            "video": segment_info["video"],
            "tag": segment_info["tag"],
            "start_time": segment_info["start_time"],
            "end_time": segment_info["end_time"],
            "caption": caption
        }
    except Exception as e:
        print(f"Error processing video {segment_info['video']}, segment {segment_info['tag']}: {e}")
        return None

def main(args):
    config = load_config()
    video_dir = config['video_data_dir']
    output_dir = config['output_dir']
    model_checkpoints_dir = config['model_checkpoints_dir']

    segments_file = os.path.join(output_dir, "video_segments.json")
    output_file = os.path.join(output_dir, "captions.json")
    model_path = os.path.join(model_checkpoints_dir, "llava-onevision-qwen2-7b-ov")
    
    device = f"cuda:{args.gpu_id}"
    device_map = {"": device}
    
    tokenizer, model, image_processor = setup_model(model_path, device_map)

    with open(segments_file, 'r', encoding='utf-8') as f:
        video_segments = json.load(f)

    # Load existing data to resume processing
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_captions = json.load(f)
        processed_keys = {(item['video'], item['tag']) for item in all_captions}
    else:
        all_captions = []
        processed_keys = set()
    
    print(f"Found {len(processed_keys)} previously processed segments. Resuming...")

    for segment in tqdm(video_segments, desc="Generating Captions"):
        if (segment['video'], segment['tag']) in processed_keys:
            continue

        video_path = os.path.join(video_dir, f"{segment['video']}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file not found, skipping: {video_path}")
            continue

        caption_data = generate_caption_for_segment(segment, video_path, model, tokenizer, image_processor, device)
        if caption_data:
            all_captions.append(caption_data)

        # Save progress periodically
        if len(all_captions) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_captions, f, indent=4, ensure_ascii=False)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_captions, f, indent=4, ensure_ascii=False)
        
    print(f"Processing complete. Captions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detailed captions for video segments.")
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU device ID to use.")
    args = parser.parse_args()
    main(args)