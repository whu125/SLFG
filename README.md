## Setup

Follow these steps to set up the project environment.

### 1. Clone Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Configure Paths

Modify the `configs/paths.json` file to match your local directory structure. You need to provide paths to your video files, datasets, model checkpoints, and desired output directory.

```json
{
    "video_data_dir": "/path/to/your/videos",
    "dataset_dir": "/path/to/your/datasets",
    "output_dir": "/path/to/project/outputs",
    "model_checkpoints_dir": "/path/to/llm/and/vision/models"
}
```

---

## Usage

Run the scripts in the specified order. The output of each script serves as the input for the next.

#### Step 1: Segment Videos

Divides each video into a fixed number of uniform time segments.

```bash
python step_1_segment_videos.py
```

#### Step 2: Generate Captions

Generates a detailed text description for each video segment created in Step 1.

```bash
python step_2_generate_captions.py --gpu-id 1
```

*Note: Adjust the `--gpu-id` parameter based on your available devices.*

#### Step 3: Extract Scenes

Breaks down the long, generated captions into smaller, more coherent scene descriptions.

```bash
python step_3_extract_scenes.py --gpu-id 0
```

#### Step 4: Generate Queries

These two scripts process the input questions to generate queries for retrieval. They can be run in parallel.

**4a. Generate Query Sentences:**

```bash
python step_4a_generate_query_sentences.py --gpu-id 2
```

**4b. Generate Query Items:**

```bash
python step_4b_generate_query_items.py --gpu-id 2
```

#### Step 5: Match Queries to Scenes

Uses a sentence-transformer model to find the most semantically relevant video scenes (from Step 3) for each query (from Step 4).

```bash
python step_5_match_queries_to_scenes.py --gpu-id 2
```

#### Step 6: Run Final Inference

Takes the top-ranked video segments identified in Step 5 and feeds them, along with the original question, into the final multimodal model to generate an answer.

```bash
python step_6_run_final_inference.py --gpu-id 2 --max-frames 32
```

*Note: Adjust `--max-frames` based on your model's requirements and GPU memory.*

---

## Project Structure

* `/configs`: Contains the `paths.json` configuration file.
* `1_segment_videos.py`: Segments videos into clips.
* `2_generate_captions.py`: Generates captions for clips.
* `3_extract_scenes.py`: Extracts scenes from captions.
* `4a_generate_query_sentences.py`: Extracts sentence queries from questions.
* `4b_generate_query_items.py`: Extracts item queries from questions.
* `5_match_queries_to_scenes.py`: Retrieves relevant scenes based on queries.
* `6_run_final_inference.py`: Generates final answer from retrieved clips.
* `README.md`: This file.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
