import os
import re
import copy
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from sample import seed_everything


TEMPLATE = """
You are an AI quality auditor for text-to-image generation.

Your task is to analyze the given image and answer a yes/no question based solely on its visual content. The question may relate to the presence of a specific object, its attributes, or relationships between multiple elements in the image.

You will also be given the original prompt used to generate the image. The prompt may provide additional context to help interpret the question, but it must never be used to supply or assume visual details.
Your judgment must rely entirely on the image itself. The image must contain clear, unmistakable visual evidence to justify a "yes" answer — the prompt cannot compensate for missing or ambiguous content.

Respond with:
- "yes" only if the answer is **clearly and unambiguously** yes based solely on the visual content. The visual evidence must be **strong, definitive, and require no assumptions or guesses**.
- "no" in **all other cases** — including if the relevant visual detail is missing, unclear, ambiguous, partially shown, obscured, or only suggested.

Even if the image closely matches what is described in the prompt, you must rely on **visible evidence** alone. If the relevant detail cannot be confirmed visually with certainty, answer "no".  
**Ambiguity equals no.**

For conditional questions, answer "yes" only if **both** the condition and the main clause are **clearly and unambiguously true** in the image. If **either part** is false or uncertain, respond "no".

Do **not** provide any explanation, justification, or extra text.  
Only return a single word: either "yes" or "no".

Example input:  
Prompt: "a golden retriever running in a grassy field under the sun"\nQuestion: "Is there a sun in the image?"  
Example output:  
"yes"

Example input:  
Prompt: "a white cat sitting on a red couch in a modern living room"  
Question: "Is the couch is present, is it red in color?"  
Example output:  
"no"
"""


class PromptImageDataset(Dataset):
    def __init__(self, image_path, image_names, eval_data):
        self.image_path = image_path
        self.eval_data = eval_data

        self.samples = []
        for name in image_names:
            key = '-'.join(name.split('-')[:3])
            if key not in eval_data:
                raise ValueError(f"Key '{key}' (from image '{name}') not found in eval_data.")
            self.samples.append((key, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, image_name = self.samples[idx]
        img_path = os.path.join(self.image_path, image_name)

        return image_name.split('.')[0], img_path, self.eval_data[key]


def custom_collate_fn(batch):
    return batch


def start_evaluation_qwen(args, mllm_path, batch_size=128):

    if "235B" in args.mllm: batch_size = int(batch_size / 2)

    llm_config = dict(
        model=mllm_path,
        max_num_seqs=batch_size,
        limit_mm_per_prompt={"image": 1, "video": 0},
        tensor_parallel_size=torch.cuda.device_count(),
        mm_encoder_tp_mode="data",
    )

    # Qwen2.5-VL Usage Guide: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html
    if "Qwen2_5" in args.mllm:
        llm_config.update(
            max_model_len=65536,
        )
    # Qwen3-VL Usage Guide: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html
    elif "Qwen3" in args.mllm:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        llm_config.update(
            max_model_len=128000,
            dtype="bfloat16",
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
        )
    llm = LLM(**llm_config)
        
    processor = AutoProcessor.from_pretrained(mllm_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.05,
        max_tokens=8192,
        stop_token_ids=[],
    )

    for MODEL in [m.strip() for m in args.model.split(",")]:
        for TASK in [t.strip() for t in args.gen_eval_file.split(",")]:
            print(f"===== Start Inference | {MODEL} | {TASK} =====")

            RESULT = {}
            image_path = os.path.join(args.output_path, MODEL, TASK)
            image_names = sorted([f for f in os.listdir(image_path) if f.lower().endswith('.png')])
            
            # If the model has already been evaluated for this task, skip the evaluation
            result_file = f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json"
            if os.path.exists(result_file) and not args.update: continue
            
            with open(f"data/{TASK.strip()}.json", 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # Prepare all inference requests in batch
            all_requests, metadata_map = [], {}
            
            dataset = PromptImageDataset(image_path, image_names, eval_data)
            
            for (ID, PATH, METADATA) in dataset:
                image_messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": PATH}],
                    }
                ]
                image_inputs, video_inputs, video_kwargs = process_vision_info(image_messages,
                    image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True,
                )
                mm_data = {"image": image_inputs}
                
                for QID, QUESTION in enumerate(METADATA["Checklist"]):

                    TEXT = f'Prompt: "{METADATA["Prompt"]}"\nQuestion: "{QUESTION["question"]}"'
                    messages = [
                        {"role": "system", "content": TEMPLATE},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": PATH},
                                {"type": "text", "text": TEXT},
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    
                    request_id = f"{ID}_{QID}"
                    all_requests.append({
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                        "request_id": request_id,
                    })
                    
                    metadata_map[request_id] = {
                        "item_id": ID,
                        "question_id": QID,
                        "metadata": METADATA
                    }
            print(f"Total requests: {len(all_requests)}")
            
            if args.update:
                with open(result_file, "r") as f: PRE_RESULT = json.load(f)
            
            for i in tqdm(range(0, len(all_requests), batch_size), desc="Batch Processing"):

                batch_requests = all_requests[i:i+batch_size]
                batch_inputs, batch_indices, outputs = [], [], [None] * len(batch_requests)

                for idx, batch_request in enumerate(batch_requests):
                    if args.update:
                        try:
                            # Only reuse previous results when both prompt and question match, and the previous result is 0/1
                            ID, QID = batch_request['request_id'].split('_')
                            assert (
                                PRE_RESULT[ID]['Checklist'][int(QID)]['question'] == batch_request['prompt'].split('Question: "')[-1].split('"<|im_end|>')[0]
                                and
                                PRE_RESULT[ID]['Prompt'] == batch_request['prompt'].split('>Prompt: "')[-1].split('"\nQuestion:')[0]
                            )
                            outputs[idx] = ["no", "yes"][PRE_RESULT[ID]['Checklist'][int(QID)]['score']]
                        except:
                            batch_inputs.append({k: v for k, v in batch_request.items() if k in ["prompt", "multi_modal_data"]})
                            batch_indices.append(idx)
                    else:
                        batch_inputs.append({k: v for k, v in batch_request.items() if k in ["prompt", "multi_modal_data"]})
                        batch_indices.append(idx)

                if batch_inputs:
                    llm_outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
                    for llm_idx, batch_idx in enumerate(batch_indices): 
                        outputs[batch_idx] = llm_outputs[llm_idx].outputs[0].text.split('</think>\n\n')[-1]

                # Process all results
                assert all(output is not None for output in outputs)
                for req, output in zip(batch_requests, outputs):
                    request_id, generated_text = req["request_id"], output
                    score = 1 if re.search(r"\byes\b", generated_text) else 0 if re.search(r"\bno\b", generated_text) else ""
                    
                    meta_info = metadata_map[request_id]
                    item_id = meta_info["item_id"]
                    question_id = meta_info["question_id"]
                    
                    if item_id not in RESULT: RESULT[item_id] = copy.deepcopy(meta_info["metadata"])
                    RESULT[item_id]['Checklist'][question_id]['score'] = score
            
            # Calculate image score for each image
            for item_id in RESULT:
                valid_scores = [item["score"] for item in RESULT[item_id]["Checklist"] if "score" in item and item["score"] in [0, 1]]
                RESULT[item_id]["image_score"] = sum(valid_scores) / len(valid_scores) if len(valid_scores) > 0 else ""
            
            # Calculate the mean score for this dimension
            mean_score_list = [meta["image_score"] for meta in RESULT.values() if meta["image_score"] != ""]
            RESULT['mean_score'] = sum(mean_score_list) / len(mean_score_list)
            
            # Save results into json
            with open(f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json", "w", encoding="utf-8") as f:
                json.dump(RESULT, f, ensure_ascii=False, indent=2)


def start_evaluation_gemini(args, mllm_path, batch_size=256, max_retries=3, timeout=30):

    import concurrent.futures
    from google import genai
    from google.genai.types import HttpOptions, Part, GenerateContentConfig
    
    for MODEL in [m.strip() for m in args.model.split(",")]:
        for TASK in [t.strip() for t in args.gen_eval_file.split(",")]:

            print(f"===== Start Inference | {MODEL} | {TASK} =====")

            # If the model has already been evaluated for this task, skip the evaluation
            result_file = f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json"
            if os.path.exists(result_file) and not args.update: continue

            RESULT = {}
            image_path = os.path.join(args.output_path, MODEL, TASK)
            image_names = sorted([f for f in os.listdir(image_path) if f.lower().endswith('.png')])
            
            with open(f"data/{TASK.strip()}.json", 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # Prepare all inference requests in batch
            all_requests, metadata_map = [], {}
            dataset = PromptImageDataset(image_path, image_names, eval_data)
            
            for (ID, PATH, METADATA) in dataset:
                with open(PATH, 'rb') as f: image_bytes = f.read()
                mm_data = Part.from_bytes(data=image_bytes, mime_type='image/png')
            
                for QID, QUESTION in enumerate(METADATA["Checklist"]):

                    TEXT = f'Prompt: "{METADATA["Prompt"]}"\nQuestion: "{QUESTION["question"]}"'

                    request_id = f"{ID}_{QID}"
                    all_requests.append({
                        "request": [mm_data, TEMPLATE + '\n' + TEXT],
                        "request_id": request_id,
                    })

                    metadata_map[request_id] = {
                        "item_id": ID,
                        "question_id": QID,
                        "metadata": METADATA
                    }
            print(f"Total requests: {len(all_requests)}")
            
            if args.update:
                with open(result_file, "r") as f: PRE_RESULT = json.load(f)

            attempt_count, requests_to_process = 1, all_requests.copy()
            while requests_to_process and attempt_count <= max_retries:

                # Init Gemini client
                client = genai.Client(api_key=args.mllm_path[1], http_options=HttpOptions(api_version="v1", timeout=1e3 * (timeout + 10 * (attempt_count - 1))))  # ms → s
                
                def call_model(input):
                    if args.update:
                        ID, QID = input['request_id'].split('_')
                        try:
                            # Only reuse previous results when both prompt and question match, and the previous result is 0/1
                            assert (
                                PRE_RESULT[ID]['Checklist'][int(QID)]['question'] == input['request'][1].split('Question: "')[-1][:-1] 
                                and
                                PRE_RESULT[ID]['Prompt'] == input['request'][1].split('\nPrompt: "')[-1].split('"\nQuestion:')[0]
                            )
                            return ["no", "yes"][int(PRE_RESULT[ID]['Checklist'][int(QID)]['score'])]
                        except:
                            pass
                    try:
                        response = client.models.generate_content(
                            model=mllm_path[0],
                            contents=input["request"],
                            config=GenerateContentConfig(temperature=0.0)
                        )
                        return response.text.strip().lower() if response and response.text else ""
                    except Exception as e:
                        print(f"[Error] request {input['request_id']}: {e}")
                        return ""

                failed_requests = []
                
                for i in tqdm(range(0, len(requests_to_process), batch_size), desc=f"Batch Processing {TASK} - Attempt {attempt_count}"):
                    batch_requests = requests_to_process[i:i+batch_size]
                    batch_inputs = batch_requests
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = list(executor.map(call_model, batch_inputs))
                        results = [{'request_id': x['request_id'], 'output': y} for (x, y) in zip (batch_inputs, futures)]

                    for req, res in zip(batch_requests, results):
                        assert req['request_id'] == res['request_id']
                        request_id, generated_text = res["request_id"], res['output']

                        if generated_text == "": failed_requests.append(req)

                        score = 1 if re.search(r"\byes\b", generated_text) else 0 if re.search(r"\bno\b", generated_text) else ""
                        meta_info = metadata_map[request_id]
                        item_id = meta_info["item_id"]
                        question_id = meta_info["question_id"]

                        if item_id not in RESULT: RESULT[item_id] = copy.deepcopy(meta_info["metadata"])
                        RESULT[item_id]['Checklist'][question_id]['score'] = score
                
                if failed_requests: print(f"Round {attempt_count}: {len(failed_requests)} requests failed, preparing to retry...")
                requests_to_process, attempt_count = failed_requests, attempt_count + 1

            if requests_to_process: print(f"Warning: {len(requests_to_process)} requests still failed after {max_retries} retries")

            # Calculate image score for each image
            for item_id in RESULT:
                valid_scores = [item["score"] for item in RESULT[item_id]["Checklist"] if "score" in item and item["score"] in [0, 1]]
                RESULT[item_id]["image_score"] = sum(valid_scores) / len(valid_scores) if len(valid_scores) > 0 else ""
            
            # Calculate the mean score for this dimension
            mean_score_list = [meta["image_score"] for meta in RESULT.values() if meta["image_score"] != ""]
            RESULT['mean_score'] = sum(mean_score_list) / len(mean_score_list)
            
            # Save results into json
            save_root = f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json"
            with open(save_root, "w", encoding="utf-8") as f:
                json.dump(RESULT, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="""
        FLUX.1-schnell, FLUX.1-dev, FLUX.1-Krea-dev | SD-3-Medium, SD-3.5-Medium, SD-3.5-Large | PixArt-Alpha, PixArt-Sigma | Qwen-Image
    """)
    parser.add_argument('--mllm', type=str, help="""
        Gemini_2_5_Flash, Qwen2_5_VL_72B, Qwen3_VL_32B_Instruct, Qwen3_VL_32B_Thinking, Qwen3_VL_30B_A3B_Instruct, Qwen3_VL_30B_A3B_Thinking, Qwen3_VL_235B_A22B_Instruct, Qwen3_VL_235B_A22B_Thinking
    """)
    parser.add_argument('--gen_eval_file', type=str, help="""
        C-MI, C-MA, C-MR, C-TR | R-LR, R-BR, R-HR, R-PR | R-GR, R-AR | R-CR, R-RR
    """)
    parser.add_argument('--output_path', type=str, default="logs")
    parser.add_argument('--update', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    seed_everything(args.seed)

    MLLMs = {
        "Gemini_2_5_Flash"       : ["gemini-2.5-flash", "GEMINI_API_KEY"],
        "Qwen2_5_VL_72B"         : "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen3_VL_32B_Instruct"  : "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen3_VL_32B_Thinking"  : "Qwen/Qwen3-VL-32B-Thinking",
        "Qwen3_VL_30B_A3B_Instruct"  : "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen3_VL_30B_A3B_Thinking"  : "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "Qwen3_VL_235B_A22B_Instruct" : "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen3_VL_235B_A22B_Thinking" : "Qwen/Qwen3-VL-235B-A22B-Thinking",
    }

    if "Qwen" in args.mllm: 
        start_evaluation_qwen(args, MLLMs[args.mllm])
    if "Gemini" in args.mllm: 
        start_evaluation_gemini(args, MLLMs[args.mllm])