import os
import re
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

def generate_query_sentence(data, model, tokenizer, device):
    """
    Generates a concise query sentence that captures the core intent of a question.
    """
    extraction_prompt_template = """
You are an expert in understanding questions. Your task is to extract the core intent from a question and formulate a concise query sentence. 
Focus ONLY on the question part, ignoring the multiple-choice options.

Example 1:
Question: "After the man with the gun threatens the cook, what does the protagonist do?"
Query: ["the man with the gun threatens the cook"]

Example 2:
Question: "Why are the mother and child, who line in front of the protagonist, unable to enter the city?"
Query: ["the mother and child are unable to enter the city"]

Now, analyze the following question:
Question: "{question}"

Expected Output in JSON format:
{{
  "query": ["Your concise query sentence here"]
}}
"""
    prompt = extraction_prompt_template.format(question=data["question"])
    messages = [{"role": "system", "content": "You are an assistant trained to extract key information."}, {"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Safely extract the query from the model's response
    try:
        match = re.search(r'\[\s*"(.*?)"\s*\]', response)
        if match:
            query_sentence = [match.group(1).strip()]
        else: # Fallback to parsing JSON
            json_response = json.loads(response)
            query_sentence = json_response.get("query", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: Could not parse LLM response for question: {data['question']}. Response: {response}")
        query_sentence = []

    return {
        "video": data["video"],
        "question": data["question"],
        "candidates": data["candidates"],
        "query_sentence": query_sentence
    }

def main(args):
    config = load_config()
    dataset_dir = config['dataset_dir']
    output_dir = config['output_dir']
    model_checkpoints_dir = config['model_checkpoints_dir']

    input_file = os.path.join(dataset_dir, "data.json")
    output_file = os.path.join(output_dir, "query_sentences.json")
    model_name = os.path.join(model_checkpoints_dir, "Qwen2.5-7B-Instruct")

    device = f"cuda:{args.gpu_id}"
    device_map = {"": device}
    model, tokenizer = setup_llm(model_name, device_map)

    with open(input_file, "r", encoding='utf-8') as f:
        dataset = json.load(f)

    output_data = [generate_query_sentence(data, model, tokenizer, device) for data in tqdm(dataset, desc="Generating Query Sentences")]

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Query sentence generation complete. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate concise query sentences from questions.")
    parser.add_argument("--gpu-id", type=int, default=2, help="GPU device ID to use.")
    args = parser.parse_args()
    main(args)