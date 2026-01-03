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


def run_inference(llm, current_requests, sampling_params, max_tokens, extract_fn, is_last_round):
    sampling_params.max_tokens = max_tokens
    request_ids, inputs = zip(*[(r['request_id'], r['input']) for r in current_requests])
    llm_outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    
    in_toks = [len(o.prompt_token_ids) for o in llm_outputs]
    out_toks = [len(o.outputs[0].token_ids) for o in llm_outputs]
    print(f"  → Token count: avg {sum(in_toks)/len(in_toks):.0f}+{sum(out_toks)/len(out_toks):.0f}, "
          f"max {max(in_toks)} + {max(out_toks)}, total {sum(in_toks):,} + {sum(out_toks):,}")

    results, retry_requests = {}, []
    for i, output in enumerate(llm_outputs):
        text = extract_fn(output.outputs[0].text)
        if text.strip() == "" and not is_last_round:
            retry_requests.append(current_requests[i])
        else:
            results[request_ids[i]] = text
    return results, retry_requests


def start_evaluation_qwen(args, mllm_path, batch_size=64, max_rounds=3, initial_max_tokens=512):

    from qwen_vl_utils import process_vision_info

    llm_config = dict(
        model=mllm_path,
        max_model_len=25600,
        max_num_seqs=batch_size,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_encoder_tp_mode="data",
    )

    # Qwen2.5-VL Usage Guide: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html
    if "Qwen2_5" in args.mllm:
        pass
    # Qwen3-VL Usage Guide: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html
    elif "Qwen3" in args.mllm:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        llm_config.update(
            dtype="bfloat16",
            enable_expert_parallel='_A' in args.mllm,  # Enable expert parallelism only for MoE models (e.g., Qwen3_VL_30B_A3B_Instruct)
            distributed_executor_backend="mp",
        )
    llm = LLM(**llm_config)
        
    processor = AutoProcessor.from_pretrained(mllm_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.05,
        stop_token_ids=[],
    )

    for MODEL in [m.strip() for m in args.model.split(",")]:
        for TASK in [t.strip() for t in args.gen_eval_file.split(",")]:

            # region [1. Pre-process: Load data and build requests]
            print(f"===== Start Inference | {MODEL} | {TASK} =====")
            image_path = os.path.join(args.output_path, MODEL, TASK)
            image_names = sorted([f for f in os.listdir(image_path) if f.lower().endswith('.png')])

            result_file = f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json"
            if os.path.exists(result_file) and not args.update: continue

            with open(f"data/{TASK.strip()}.json", 'r', encoding='utf-8') as f:
                eval_data = json.load(f)

            # Load cached results if updating
            PRE_RESULT = json.load(open(result_file, "r")) if args.update and os.path.exists(result_file) else None

            all_requests, metadata_map, cached_results = [], {}, {}
            dataset = PromptImageDataset(image_path, image_names, eval_data)

            for (ID, PATH, METADATA) in dataset:
                # Check if result can be reused from cache
                def is_cached(qid, question):
                    try:
                        return (
                            PRE_RESULT[ID]['Prompt'] == METADATA["Prompt"] and
                            PRE_RESULT[ID]['Checklist'][qid]['question'] == question["question"] and
                            PRE_RESULT[ID]['Checklist'][qid].get('score') in [0, 1]
                        )
                    except: 
                        return False
                
                # Skip image preprocessing if all questions are cached
                if all(is_cached(qid, q) for qid, q in enumerate(METADATA["Checklist"])):
                    for qid, q in enumerate(METADATA["Checklist"]):
                        request_id = f"{ID}_{qid}"
                        cached_results[request_id] = ["no", "yes"][PRE_RESULT[ID]['Checklist'][qid]['score']]
                        metadata_map[request_id] = {"item_id": ID, "question_id": qid, "metadata": METADATA}
                    continue
                
                # Preprocess image
                image_inputs, _, _ = process_vision_info(
                    [{"role": "user", "content": [{"type": "image", "image": PATH}]}],
                    image_patch_size=processor.image_processor.patch_size, 
                    return_video_kwargs=True, return_video_metadata=True
                )
                mm_data = {"image": image_inputs}
                
                # Create request for each question
                for QID, QUESTION in enumerate(METADATA["Checklist"]):
                    request_id = f"{ID}_{QID}"
                    metadata_map[request_id] = {"item_id": ID, "question_id": QID, "metadata": METADATA}
                    
                    # Check individual question cache
                    if is_cached(QID, QUESTION):
                        cached_results[request_id] = ["no", "yes"][PRE_RESULT[ID]['Checklist'][QID]['score']]
                        continue
                    
                    TEXT = f'Prompt: "{METADATA["Prompt"]}"\nQuestion: "{QUESTION["question"]}"'
                    messages = [
                        {"role": "system", "content": TEMPLATE},
                        {"role": "user", "content": [{"type": "image", "image": PATH}, {"type": "text", "text": TEXT}]}
                    ]
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    all_requests.append({
                        "request_id": request_id,
                        "input": {"prompt": prompt, "multi_modal_data": mm_data}
                    })

            print(f"Total: {len(all_requests)} need inference, {len(cached_results)} cached")
            # endregion

            # region [2. Evaluation: Multi-round inference]
            def extract_fn(text):
                if 'thinking' in args.mllm.lower():
                    return text.split('</think>\n\n')[-1] if '</think>\n\n' in text else ""
                return text

            current_requests, current_max_tokens = all_requests, initial_max_tokens
            for round_idx in range(max_rounds):
                if not current_requests: break
                print(f"[Round {round_idx+1}/{max_rounds}] {len(current_requests)} requests, max_tokens={current_max_tokens}")
                
                results, current_requests = run_inference(
                    llm, current_requests, sampling_params, 
                    current_max_tokens, extract_fn, round_idx == max_rounds - 1
                )
                cached_results.update(results)
                current_max_tokens = min(current_max_tokens * 4, 8192)
            # endregion

            # region [3. Post-process: Save the results]
            RESULT = {}
            for request_id in sorted(cached_results.keys()):
                generated_text = cached_results[request_id]
                score = 1 if re.search(r"\byes\b", generated_text) else 0 if re.search(r"\bno\b", generated_text) else ""
                meta = metadata_map[request_id]
                if meta["item_id"] not in RESULT: 
                    RESULT[meta["item_id"]] = copy.deepcopy(meta["metadata"])
                    for q in RESULT[meta["item_id"]]['Checklist']: q['score'] = ""
                RESULT[meta["item_id"]]['Checklist'][meta["question_id"]]['score'] = score

            for item_id in RESULT:
                valid_scores = [item["score"] for item in RESULT[item_id]["Checklist"] if "score" in item and item["score"] in [0, 1]]
                RESULT[item_id]["image_score"] = sum(valid_scores) / len(valid_scores) if len(valid_scores) > 0 else ""

            mean_score_list = [meta["image_score"] for meta in RESULT.values() if meta["image_score"] != ""]
            RESULT['mean_score'] = sum(mean_score_list) / len(mean_score_list) if len(mean_score_list) > 0 else ""

            with open(f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json", "w", encoding="utf-8") as f:
                json.dump(RESULT, f, ensure_ascii=False, indent=2)
            # endregion


def start_evaluation_gemini(args, mllm_path, batch_size=256, max_retries=3, timeout=30):

    import concurrent.futures
    from google import genai
    from google.genai.types import HttpOptions, Part, GenerateContentConfig
    
    for MODEL in [m.strip() for m in args.model.split(",")]:
        for TASK in [t.strip() for t in args.gen_eval_file.split(",")]:

            # region [1. Pre-process: Load data and build requests]
            print(f"===== Start Inference | {MODEL} | {TASK} =====")
            image_path = os.path.join(args.output_path, MODEL, TASK)
            image_names = sorted([f for f in os.listdir(image_path) if f.lower().endswith('.png')])

            result_file = f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json"
            if os.path.exists(result_file) and not args.update: continue

            with open(f"data/{TASK.strip()}.json", 'r', encoding='utf-8') as f:
                eval_data = json.load(f)

            # Load cached results if updating
            PRE_RESULT = json.load(open(result_file, "r")) if args.update and os.path.exists(result_file) else None

            all_requests, metadata_map, cached_results = [], {}, {}
            dataset = PromptImageDataset(image_path, image_names, eval_data)
            
            for (ID, PATH, METADATA) in dataset:
                # Check if result can be reused from cache
                def is_cached(qid, question):
                    try:
                        return (
                            PRE_RESULT[ID]['Prompt'] == METADATA["Prompt"] and
                            PRE_RESULT[ID]['Checklist'][qid]['question'] == question["question"] and
                            PRE_RESULT[ID]['Checklist'][qid].get('score') in [0, 1]
                        )
                    except: 
                        return False
                
                # Skip image preprocessing if all questions are cached
                if all(is_cached(qid, q) for qid, q in enumerate(METADATA["Checklist"])):
                    for qid, q in enumerate(METADATA["Checklist"]):
                        request_id = f"{ID}_{qid}"
                        cached_results[request_id] = ["no", "yes"][PRE_RESULT[ID]['Checklist'][qid]['score']]
                        metadata_map[request_id] = {"item_id": ID, "question_id": qid, "metadata": METADATA}
                    continue
                
                # Preprocess image
                with open(PATH, 'rb') as f: image_bytes = f.read()
                mm_data = Part.from_bytes(data=image_bytes, mime_type='image/png')
                
                # Create request for each question
                for QID, QUESTION in enumerate(METADATA["Checklist"]):
                    request_id = f"{ID}_{QID}"
                    metadata_map[request_id] = {"item_id": ID, "question_id": QID, "metadata": METADATA}
                    
                    # Check individual question cache
                    if is_cached(QID, QUESTION):
                        cached_results[request_id] = ["no", "yes"][PRE_RESULT[ID]['Checklist'][QID]['score']]
                        continue
                    
                    TEXT = f'Prompt: "{METADATA["Prompt"]}"\nQuestion: "{QUESTION["question"]}"'
                    all_requests.append({
                        "request": [mm_data, TEMPLATE + '\n' + TEXT],
                        "request_id": request_id,
                    })

            print(f"Total: {len(all_requests)} need inference, {len(cached_results)} cached")
            # endregion

            # region [2. Evaluation: Multi-round inference]
            attempt_count, current_requests = 1, all_requests
            while current_requests and attempt_count <= max_retries:
                # Init Gemini client with dynamic timeout
                client = genai.Client(
                    api_key=mllm_path[1], 
                    http_options=HttpOptions(api_version="v1", timeout=1e3 * (timeout + 10 * (attempt_count - 1)))  # s → ms
                )
                
                def call_model(request):
                    try:
                        response = client.models.generate_content(
                            model=mllm_path[0],
                            contents=request["request"],
                            config=GenerateContentConfig(temperature=0.0)
                        )
                        return response.text.strip().lower() if response and response.text else ""
                    except Exception as e:
                        print(f"[Error] request {request['request_id']}: {e}")
                        return ""

                failed_requests = []
                for i in tqdm(range(0, len(current_requests), batch_size), desc=f"Batch Processing {TASK} - Attempt {attempt_count}"):
                    batch_requests = current_requests[i:i+batch_size]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = list(executor.map(call_model, batch_requests))

                    for req, generated_text in zip(batch_requests, futures):
                        if generated_text == "": 
                            failed_requests.append(req)
                            continue
                        cached_results[req['request_id']] = generated_text
                
                if failed_requests: print(f"Round {attempt_count}: {len(failed_requests)} requests failed, preparing to retry...")
                current_requests, attempt_count = failed_requests, attempt_count + 1 

            if current_requests: print(f"Warning: {len(current_requests)} requests still failed after {max_retries} retries")
            # endregion

            # region [3. Post-process: Save the results]
            RESULT = {}
            for request_id in sorted(cached_results.keys()):
                generated_text = cached_results[request_id]
                score = 1 if re.search(r"\byes\b", generated_text) else 0 if re.search(r"\bno\b", generated_text) else ""
                meta = metadata_map[request_id]
                if meta["item_id"] not in RESULT: 
                    RESULT[meta["item_id"]] = copy.deepcopy(meta["metadata"])
                    for q in RESULT[meta["item_id"]]['Checklist']: q['score'] = ""
                RESULT[meta["item_id"]]['Checklist'][meta["question_id"]]['score'] = score

            for item_id in RESULT:
                valid_scores = [item["score"] for item in RESULT[item_id]["Checklist"] if "score" in item and item["score"] in [0, 1]]
                RESULT[item_id]["image_score"] = sum(valid_scores) / len(valid_scores) if len(valid_scores) > 0 else ""

            mean_score_list = [meta["image_score"] for meta in RESULT.values() if meta["image_score"] != ""]
            RESULT['mean_score'] = sum(mean_score_list) / len(mean_score_list) if len(mean_score_list) > 0 else ""

            with open(f"{args.output_path}/{MODEL}/{TASK}-{args.mllm}.json", "w", encoding="utf-8") as f:
                json.dump(RESULT, f, ensure_ascii=False, indent=2)
            # endregion


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="""
        FLUX.1-schnell, FLUX.1-dev, FLUX.1-Krea-dev | SD-3-Medium, SD-3.5-Medium, SD-3.5-Large | PixArt-Alpha, PixArt-Sigma | Qwen-Image
    """)
    parser.add_argument('--mllm', type=str, help="""
        Qwen-series: Qwen2_5_VL_72B, Qwen3_VL_8B_Instruct, Qwen3_VL_8B_Thinking, Qwen3_VL_32B_Instruct, Qwen3_VL_32B_Thinking, Qwen3_VL_30B_A3B_Instruct, Qwen3_VL_30B_A3B_Thinking, Qwen3_VL_235B_A22B_Instruct, Qwen3_VL_235B_A22B_Thinking
        Gemini-series: Gemini_2_5_Flash
    """)
    parser.add_argument('--gen_eval_file', type=str, help="""
        Composition: C-MI, C-MA, C-MR, C-TR | Reasoning: R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR
    """)
    parser.add_argument('--output_path', type=str, default="logs")
    parser.add_argument('--update', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    seed_everything(args.seed)

    MLLMs = {
        "Qwen2_5_VL_72B"              : "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen3_VL_8B_Instruct"        : "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3_VL_8B_Thinking"        : "Qwen/Qwen3-VL-8B-Thinking",
        "Qwen3_VL_32B_Instruct"       : "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen3_VL_32B_Thinking"       : "Qwen/Qwen3-VL-32B-Thinking",
        "Qwen3_VL_30B_A3B_Instruct"   : "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen3_VL_30B_A3B_Thinking"   : "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "Qwen3_VL_235B_A22B_Instruct" : "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "Qwen3_VL_235B_A22B_Thinking" : "Qwen/Qwen3-VL-235B-A22B-Thinking",
        "Gemini_2_5_Flash"            : ["gemini-2.5-flash", os.getenv("GEMINI_API_KEY")],
    }
    
    if "Qwen" in args.mllm: 
        start_evaluation_qwen(args, MLLMs[args.mllm])
    if "Gemini" in args.mllm: 
        start_evaluation_gemini(args, MLLMs[args.mllm])