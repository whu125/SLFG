import os
import json
import ffmpeg
from tqdm import tqdm
import argparse

def load_config(config_path='configs/paths.json'):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_videos(video_folder, output_file, num_segments=32):
    """
    Divides each video in a folder into a fixed number of time segments
    and saves the segment information to a JSON file.

    Args:
        video_folder (str): Path to the directory containing video files.
        output_file (str): Path to save the output JSON file.
        num_segments (int): The number of segments to divide each video into.
    """
    video_segments = []
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos to process.")

    for filename in tqdm(video_files, desc="Segmenting Videos"):
        video_path = os.path.join(video_folder, filename)
        video_name = os.path.splitext(filename)[0]

        try:
            # Get video duration using ffmpeg
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            segment_duration = duration / num_segments

            for i in range(num_segments):
                start_time = round(i * segment_duration, 3)
                end_time = round((i + 1) * segment_duration, 3)
                video_segments.append({
                    "video": video_name,
                    "tag": i + 1,
                    "start_time": start_time,
                    "end_time": end_time,
                })
        except ffmpeg.Error as e:
            print(f"Error processing {filename}: {e.stderr.decode('utf8')}")
            continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(video_segments, f, indent=4, ensure_ascii=False)

    print(f"Video segment information saved to {output_file}")

def main():
    config = load_config()
    video_folder = config['video_data_dir']
    output_dir = config['output_dir']
    
    # Define the output path for the segment data
    output_file = os.path.join(output_dir, "video_segments.json")
    segment_videos(video_folder, output_file)

if __name__ == "__main__":
    main()