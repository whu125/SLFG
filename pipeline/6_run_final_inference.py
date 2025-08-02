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

def extract_subtitles(video_path, subtitle_path):
    print(f"Extracting subtitles for video: {video_path}")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    print(f"Total frames in video: {total_frame}, extracted {len(subtitle_frames)} subtitle frames.")
    return subtitle_frames, total_frame

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(subtitle_path):
    subtitles = {}
    print(f"Loading subtitles from {subtitle_path}")
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    print(f"Loaded {len(subtitles)} subtitle entries.")
    return subtitles

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        print(f"Extracting audio from {video_path} to {audio_path}")
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()

def chunk_audio(audio_path, chunk_length_s=30):
    print(f"Chunking audio: {audio_path}, chunk length: {chunk_length_s}s")
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)  
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)  
    num_samples_per_chunk = chunk_length_s * 16000 
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    print(f"Chunked audio into {len(chunks)} parts.")
    return chunks

def transcribe_chunk(chunk):
    print(f"Transcribing audio chunk...")
    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Transcription completed: {transcription[:50]}...")  # Print first 50 chars
    return transcription

def get_asr_docs(video_path, audio_path):
    print(f"Getting ASR docs for video: {video_path}")
    full_transcription = []
    try:
        extract_audio(video_path, audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    print(f"Transcription completed for {len(full_transcription)} chunks.")
    return full_transcription

def get_ocr_docs(frames):
    print(f"Performing OCR on {len(frames)} frames.")
    reader = easyocr.Reader(['en']) 
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    print(f"OCR completed with {len(ocr_docs)} documents.")
    return ocr_docs

def save_frames(frames):
    print(f"Saving {len(frames)} frames to disk.")
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    print(f"Saved {len(file_paths)} frames.")
    return file_paths
    
def get_det_docs(frames, prompt):
    print(f"Getting DET docs for frames: {len(frames)} with prompt: {prompt[:30]}...")  # Print first 30 chars
    prompt = ",".join(prompt)
    frames_path = save_frames(frames)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except Exception as e:
            print(f"Error in DET docs retrieval: {e}")
            res = []
    print(f"DET docs retrieval completed with {len(res)} results.")
    return res

def det_preprocess(det_docs, location, relation, number):
    scene_descriptions = []
    print(f"Preprocessing DET docs for {len(det_docs)} entries.")
    return scene_descriptions

def load_config(config_path='configs/paths.json'):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_inference_model(model_path, device_map):
    """Loads the multimodal model for final inference."""
    print(f"Loading inference model from: {model_path}")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, torch_dtype="bfloat16",
        attn_implementation="sdpa", device_map=device_map
    )
    model.eval()
    return tokenizer, model, image_processor

def process_video_segments(video_path, time_segments, max_frames, frames_per_segment=8):
    """
    Extracts frames from multiple specified time segments in a video.
    Pads if the total number of frames is less than max_frames.
    """
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
        if start_time >= end_time:
            continue
            
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        segment_frame_indices = np.linspace(start_frame, end_frame, frames_per_segment, dtype=int)
        segment_frame_indices = [min(f, total_frames_in_video - 1) for f in segment_frame_indices]
        
        try:
            frames = vr.get_batch(segment_frame_indices).asnumpy()
            all_extracted_frames.extend(frames)
        except Exception as e:
            print(f"Error extracting frames from segment {start_time}-{end_time}: {e}")
            
    if not all_extracted_frames:
        return np.array([])
        
    final_frames = np.array(all_extracted_frames)
    # Pad with the last frame if necessary
    if len(final_frames) < max_frames:
        padding_needed = max_frames - len(final_frames)
        last_frame = final_frames[-1]
        padding = np.array([last_frame] * padding_needed)
        final_frames = np.concatenate((final_frames, padding), axis=0)
        
    return final_frames[:max_frames]

def run_llava_inference(question, candidates, video_frames, model, tokenizer, image_processor, device):
    """Runs inference with the LLaVA model on the given frames and question."""
    video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(device).bfloat16()
    
    # Construct a multiple-choice prompt
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
    
    input_ids = tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=10, # Short response expected
        )
    
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

def main(args):
    config = load_config()
    video_dir = config['video_data_dir']
    dataset_dir = config['dataset_dir']
    output_dir = config['output_dir']
    model_checkpoints_dir = config['model_checkpoints_dir']

    matches_file = os.path.join(output_dir, "matches.json")
    original_dataset_file = os.path.join(dataset_dir, "data.json")
    
    output_results_file = os.path.join(output_dir, f"inference_results_{args.max_frames}frames.json")
    output_errors_file = os.path.join(output_dir, f"inference_errors_{args.max_frames}frames.json")
    model_path = os.path.join(model_checkpoints_dir, "llava-onevision-qwen2-7b-ov")
    
    device = f"cuda:{args.gpu_id}"
    device_map = {"": device}
    tokenizer, model, image_processor = setup_inference_model(model_path, device_map)
    
    with open(matches_file, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    with open(original_dataset_file, 'r', encoding='utf-8') as f:
        original_data = {item['question']: item for item in json.load(f)}

    results = []
    errors = []

    for item in tqdm(match_data, desc="Running Final Inference"):
        video_name = item['video']
        question = item['question']
        
        time_segments = []
        for i in range(1, 5):
            if f'start_time_{i}' in item:
                time_segments.append({'start': item[f'start_time_{i}'], 'end': item[f'end_time_{i}']})
        
        if not time_segments:
            errors.append({'video': video_name, 'question': question, 'error': 'No time segments from matching step.'})
            continue

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            errors.append({'video': video_name, 'question': question, 'error': 'Video file not found.'})
            continue
            
        try:
            frames = process_video_segments(video_path, time_segments, args.max_frames)
            if frames.size == 0:
                raise ValueError("Frame processing returned no frames.")
            
            response = run_llava_inference(
                question, item['candidates'], frames, model, tokenizer, image_processor, device
            )
            
            original_item = original_data.get(question)
            
            results.append({
                'video': video_name,
                'question_id': original_item.get('question_id') if original_item else None,
                'question': question,
                'candidates': item['candidates'],
                'ground_truth_answer': original_item.get('answer') if original_item else None,
                'predicted_answer': response
            })
            
            if len(results) % 5 == 0:
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
    parser = argparse.ArgumentParser(description="Run final inference on retrieved video segments.")
    parser.add_argument("--gpu-id", type=int, default=2, help="GPU device ID for inference.")
    parser.add_argument("--max-frames", type=int, default=32, help="Total number of frames to feed into the model.")
    args = parser.parse_args()
    main(args)