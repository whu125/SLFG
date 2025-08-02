import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_config(config_path='configs/paths.json'):
    """Loads the configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_query_files(sentence_file, item_file):
    """
    Merges query sentence and query item files based on the 'question' key.
    """
    print("--- Step 1: Merging Query Files ---")
    with open(sentence_file, "r", encoding='utf-8') as f1:
        data_sentence = json.load(f1)
    with open(item_file, "r", encoding='utf-8') as f2:
        data_item = json.load(f2)

    item_lookup = {item.get("question"): item for item in data_item}
    
    combined_data = []
    for sentence_item in data_sentence:
        question = sentence_item.get("question")
        if question in item_lookup:
            item_item = item_lookup[question]
            combined_data.append({
                "video": sentence_item.get("video"),
                "question": question,
                "candidates": sentence_item.get("candidates"),
                "query_sentence": sentence_item.get("query_sentence", []),
                "query_item": item_item.get("query_item", [])
            })

    print(f"Merging complete. Found {len(combined_data)} matching query items.")
    return combined_data

def match_queries_to_scenes(queries, scenes, model, device, top_k=4):
    """
    Finds the best matching video scenes for each query using cosine similarity.
    """
    print("\n--- Step 2: Matching Queries to Scenes ---")
    all_results = []
    error_questions = []

    # Pre-process scenes into a dictionary for efficient lookup
    scenes_by_video = {}
    for scene in scenes:
        vid = scene.get('video')
        if vid not in scenes_by_video:
            scenes_by_video[vid] = []
        scenes_by_video[vid].append(scene)

    for query in tqdm(queries, desc="Matching Queries"):
        try:
            vid = query.get('video')
            question = query.get('question')
            
            query_text = " ".join(query.get('query_sentence', []) + query.get('query_item', []))
            if not query_text.strip():
                error_questions.append(question)
                continue

            query_embedding = model.encode(query_text, device=device, normalize_embeddings=True)
            
            relevant_scenes = scenes_by_video.get(vid)
            if not relevant_scenes:
                error_questions.append(question)
                continue

            # Find the best similarity score for each unique scene tag
            tag_scores = {}
            for scene in relevant_scenes:
                scene_texts = [s for s in scene.get('scenes', []) if isinstance(s, str) and s.strip()]
                if not scene_texts:
                    continue

                scene_embeddings = model.encode(scene_texts, device=device, normalize_embeddings=True)
                similarities = cosine_similarity([query_embedding], scene_embeddings)[0]
                
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]
                tag = scene.get('tag')
                
                if tag not in tag_scores or max_sim > tag_scores[tag][0]:
                    tag_scores[tag] = (float(max_sim), scene.get('start_time'), scene.get('end_time'))

            if not tag_scores:
                error_questions.append(question)
                continue
            
            # Sort matches by similarity and get top K
            sorted_matches = sorted(tag_scores.items(), key=lambda item: item[1][0], reverse=True)
            top_results = sorted_matches[:top_k]

            result = {'video': vid, 'question': question, **query}
            for i, (tag, (sim, start, end)) in enumerate(top_results):
                result[f'best_tag_{i+1}'] = tag
                result[f'best_similarity_{i+1}'] = sim
                result[f'start_time_{i+1}'] = start
                result[f'end_time_{i+1}'] = end
            
            all_results.append(result)

        except Exception as e:
            print(f"\nAn unexpected error occurred with question '{query.get('question')}': {e}")
            error_questions.append(query.get('question'))

    return all_results, list(set(error_questions))

def main(args):
    config = load_config()
    output_dir = config['output_dir']
    model_checkpoints_dir = config['model_checkpoints_dir']

    query_sentence_file = os.path.join(output_dir, "query_sentences.json")
    query_item_file = os.path.join(output_dir, "query_items.json")
    scene_file = os.path.join(output_dir, "scenes.json")
    
    output_match_file = os.path.join(output_dir, "matches.json")
    output_error_file = os.path.join(output_dir, "matches_errors.json")
    
    model_path = os.path.join(model_checkpoints_dir, "bge-m3")
    
    combined_queries = merge_query_files(query_sentence_file, query_item_file)
    if not combined_queries:
        print("Halting: No queries to process after merging.")
        return

    with open(scene_file, 'r', encoding='utf-8') as f:
        scenes_data = json.load(f)

    device = f'cuda:{args.gpu_id}'
    print(f"\nUsing device: {device}")
    model = SentenceTransformer(model_path, device=device)
    
    matched_results, error_qids = match_queries_to_scenes(combined_queries, scenes_data, model, device)

    print("\n--- Step 3: Saving Results ---")
    if matched_results:
        print(f"Saving {len(matched_results)} matched results to {output_match_file}")
        with open(output_match_file, 'w', encoding='utf-8') as f:
            json.dump(matched_results, f, indent=4, ensure_ascii=False)

    if error_qids:
        print(f"\nFound {len(error_qids)} errors. Saving error log to {output_error_file}")
        with open(output_error_file, 'w', encoding='utf-8') as f:
            json.dump({"error_question_ids": error_qids}, f, indent=4)

    print("\nProcessing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match queries to video scenes using semantic similarity.")
    parser.add_argument("--gpu-id", type=int, default=2, help="GPU device ID for SentenceTransformer model.")
    args = parser.parse_args()
    main(args)