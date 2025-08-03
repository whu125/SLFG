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


def setup_inference_model(model_path, device_map):
    """Loads the multimodal model for final inference."""
    print(f"Loading inference model: {model_path}")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, torch_dtype="bfloat16",
        attn_implementation="sdpa", device_map=device_map
    )
    model.eval()
    return tokenizer, model, image_processor


def reorganize_scenes(item, score_threshold, max_frames, frames_per_segment, delta_t, video_durations):
    """
    [Integrated Function] Reorganizes scenes and allocates frames for a single item.
    This function performs the reorganization logic in memory.
    """
    # 1. Extract and sort groups (input data is pre-sorted)
    groups = []
    for i in range(1, 5):
        if f'best_similarity_{i}' in item:
            groups.append({
                'score': item[f'best_similarity_{i}'],
                'start': item[f'start_time_{i}'],
                'end': item[f'end_time_{i}'],
            })
    if not groups:
        return []

    # 2. Apply score threshold to filter groups
    selected_groups = [groups[0]]
    for i in range(1, len(groups)):
        score_diff = selected_groups[-1]['score'] - groups[i]['score']
        if score_diff > score_threshold:
            break
        selected_groups.append(groups[i])

    # 3. Manage frame budget and extend context
    num_selected_groups = len(selected_groups)
    total_initial_frames = num_selected_groups * frames_per_segment
    final_segments = []

    if total_initial_frames > max_frames:
        frames_per_group_after_cut = max_frames // num_selected_groups
        for group in selected_groups:
            final_segments.append({
                "start": group['start'],
                "end": group['end'],
                "frames_to_sample": frames_per_group_after_cut
            })
    else:
        remaining_frame_budget = max_frames - total_initial_frames
        base_extra_frames = remaining_frame_budget // num_selected_groups
        remainder_frames = remaining_frame_budget % num_selected_groups

        for i, group in enumerate(selected_groups):
            group_extra_frames = base_extra_frames + (1 if i < remainder_frames else 0)
            total_frames_for_group = frames_per_segment + group_extra_frames

            original_duration = group['end'] - group['start']
            if original_duration <= 0: original_duration = delta_t

            time_per_original_frame = original_duration / frames_per_segment
            time_extension = group_extra_frames * time_per_original_frame

            new_start = group['start'] - time_extension / 2
            new_end = group['end'] + time_extension / 2

            video_duration = video_durations.get(item['video']) if video_durations else None
            new_start = max(0, new_start)
            if video_duration:
                new_end = min(new_end, video_duration)

            final_segments.append({
                "start": new_start,
                "end": new_end,
                "frames_to_sample": total_frames_for_group
            })
    return final_segments


def process_video_segments(video_path, time_segments):
    """Extracts a specified number of frames from multiple time segments based on a plan."""
    try:
        vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    except Exception as e:
        print(f"Error opening video {video_path}: {e}")
        return np.array([])

    fps = vr.get_avg_fps()
    total_frames_in_video = len(vr)
    all_extracted_frames = []

    for segment in time_segments:
        start_time = float(segment.get('start', 0))
        end_time = float(segment.get('end', 0))
        frames_to_sample = int(segment.get('frames_to_sample', 0))

        if start_time >= end_time or frames_to_sample <= 0:
            continue

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        if start_frame > end_frame: start_frame = end_frame

        segment_frame_indices = np.linspace(start_frame, end_frame, frames_to_sample, dtype=int)
        segment_frame_indices = [min(f, total_frames_in_video - 1) for f in segment_frame_indices]

        try:
            frames = vr.get_batch(segment_frame_indices).asnumpy()
            all_extracted_frames.extend(frames)
        except Exception as e:
            print(f"Error extracting frames from segment {start_time}-{end_time}: {e}")

    if not all_extracted_frames:
        return np.array([])

    return np.array(all_extracted_frames)


def run_llava_inference(question, candidates, video_frames, model, tokenizer, image_processor, device):
    """Runs LLaVA model inference on the given frames and question."""
    video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(device).bfloat16()

    options_str = " ".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(candidates)])
    prompt_text = (
        f"Select the best answer to the following multiple-choice question based on the video. "
        f"Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
        f"Question: {question}\n{options_str}"
    )

    full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_for_model = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)

    with torch.no_grad():
        cont = model.generate(
            input_ids, images=[video_tensor], modalities=["video"],
            do_sample=False, max_new_tokens=10,
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()


def main(args):
    # Paths are now taken directly from command-line arguments
    video_dir = args.video_dir
    matches_file = args.matches_file
    original_dataset_file = args.original_dataset_file
    output_dir = args.output_dir
    model_path = args.model_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_results_file = os.path.join(output_dir, f"inference_results_reorg_{args.max_frames}frames.json")
    output_errors_file = os.path.join(output_dir, f"inference_errors_reorg.json")

    device = f"cuda:{args.gpu_id}"
    device_map = {"": device}
    tokenizer, model, image_processor = setup_inference_model(model_path, device_map)

    with open(matches_file, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    # Load original dataset and create a lookup table based on the question text
    with open(original_dataset_file, 'r', encoding='utf-8') as f:
        original_data_list = json.load(f)
    original_data_lookup = {item['question']: item for item in original_data_list}

    results = []
    errors = []

    # (Simulation) Get video durations, may need real data in a real scenario
    video_durations = {}

    for item in tqdm(match_data, desc="Performing reorganization and inference"):
        video_name = item['video']
        question = item['question']

        # === CHANGE HIGHLIGHT: Get candidates from the metadata lookup table ===
        original_item = original_data_lookup.get(question)
        if not original_item:
            errors.append(
                {'video': video_name, 'question': question, 'error': 'Metadata not found in original dataset file.'})
            continue

        candidates = original_item.get('candidates')
        if not candidates:
            errors.append({'video': video_name, 'question': question, 'error': 'Candidates not found in metadata.'})
            continue
        # =======================================================================

        # Step 1: Perform scene reorganization in memory
        time_segments = reorganize_scenes(
            item, args.threshold, args.max_frames,
            args.frames_per_segment, args.delta_t, video_durations
        )

        if not time_segments:
            errors.append(
                {'video': video_name, 'question': question, 'error': 'No time segments left after reorganization.'})
            continue

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            video_path_mkv = os.path.join(video_dir, f"{video_name}.mkv")
            if not os.path.exists(video_path_mkv):
                errors.append(
                    {'video': video_name, 'question': question, 'error': 'Video file not found (.mp4 or .mkv).'})
                continue
            video_path = video_path_mkv

        try:
            # Step 2: Extract video frames
            frames = process_video_segments(video_path, time_segments)
            if frames.size == 0:
                raise ValueError("Frame processing returned no frames.")

            # Step 3: Run model inference using candidates from metadata
            response = run_llava_inference(
                question, candidates, frames, model, tokenizer, image_processor, device
            )

            results.append({
                'video': video_name,
                'question_id': original_item.get('question_id'),
                'question': question,
                'candidates': candidates,
                'ground_truth_answer': original_item.get('answer'),
                'predicted_answer': response
            })
            print(f"ground_truth_answer {original_item.get('answer')} ,predicted_answer: {response})")
            if len(results) % 10 == 0:
                with open(output_results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing video {video_name} (question: {question}): {e}")
            errors.append({'video': video_name, 'question': question, 'error': str(e)})

    with open(output_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    with open(output_errors_file, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)

    print(f"\nInference finished. Results saved to {output_results_file}.")
    if errors:
        print(f"Encountered {len(errors)} errors. Log saved to {output_errors_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run final inference with reorganization logic on retrieved video segments.")
    # --- Paths ---
    parser.add_argument("--matches-file", type=str, required=True, help="Path to the input matches JSON file.")
    parser.add_argument("--video-dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--original-dataset-file", type=str, required=True,
                        help="Path to the original dataset JSON file for metadata.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the output files.")

    # --- Parameters ---
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for inference.")
    parser.add_argument("--max-frames", type=int, default=32, help="Target total number of frames for the model.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Score difference threshold for selecting groups.")
    parser.add_argument("--frames-per-segment", type=int, default=8, help="Initial number of frames sampled per scene.")
    parser.add_argument("--delta-t", type=float, default=1.0,
                        help="Assumed sampling interval for a scene if its duration is zero.")
    args = parser.parse_args()
    main(args)
