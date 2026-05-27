import json
import os
import re
import time
import torch
import gc
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

DOMAIN_MODELS = {
    "Surgery": "./EgoCross-main/models/surgery",
    "Animal": "./EgoCross-main/models/animal",
    "Industry": "./EgoCross-main/models/industry",
    "XSports": "./EgoCross-main/models/xsports"
}

DATASET_TO_DOMAIN = {
    "CholecTrack20": "Surgery", "EgoSurgery": "Surgery",
    "EgoPet": "Animal", "ENIGMA": "Industry", "ExtrameSportFPV": "XSports"
}

def fix_image_paths(paths):
    if isinstance(paths, str):
        return paths.replace("/egocross_testbed/", "./EgoCross-main/datasets/egocross_testbed/")
    return [p.replace("/egocross_testbed/", "./EgoCross-main/datasets/egocross_testbed/") for p in paths]

def load_domain_model(model_path):
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        min_pixels=50176,
        max_pixels=360000
    )
    return model, processor

def inference_agent(model, processor, prompt, video_path=None, start_frame_index=0, sampling_fps=0.5):
    messages = []
    
    if video_path:
        content_list = []
        if isinstance(video_path, list):
            interval = 1.0 / sampling_fps if sampling_fps > 0 else 2.0 
            
            for i, path in enumerate(video_path):
                current_sec = (start_frame_index + i) * interval
                content_list.append({"type": "text", "text": f"{current_sec:g}s\n"})
                content_list.append({"type": "image", "image": path, "max_pixels": 360000})

            
            content_list.append({"type": "text", "text": prompt})
        else:
            content_list.append({
                "type": "video", 
                "video": video_path, 
                "max_pixels": 360000
            })
            content_list.append({"type": "text", "text": prompt})
            
        messages = [{"role": "user", "content": content_list}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    if "fps" in video_kwargs:
        video_kwargs.pop("fps")
            
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs
    ).to("cuda:0")
    
    rep_penalty = 1.05
    generated_ids = model.generate(**inputs, max_new_tokens=2048, repetition_penalty=rep_penalty)
    input_length = inputs.input_ids.shape[1]
    response = processor.batch_decode([out_ids[input_length:] for out_ids in generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    del inputs, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()
    
    return response

def parse_answer(response: str):
    prediction = ""
    try: prediction = json.loads(response).get("prediction", "")
    except:
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            try: prediction = json.loads(json_match.group(0)).get("prediction", "")
            except: pass
    if not prediction:
        match = re.search(r'\b([A-D])\b', response)
        prediction = match.group(1) if match else "A"
    return prediction.strip().upper()

def main():
    testbed_path = "datasets/egocross_testbed_imgs.json" if os.path.exists("datasets/egocross_testbed_imgs.json") else "egocross_testbed_imgs.json"
    with open(testbed_path, "r", encoding="utf-8") as f: testbed_data = json.load(f)
    with open("submission_template.json", "r", encoding="utf-8") as f: submission_data = json.load(f)
        
    surgery_tasks = []
    testbed_dict = {str(item.get("question_id", item.get("id", ""))): item for item in testbed_data if len(str(item.get("question_id", item.get("id", "")))) > 5}
            
    for i, sub_item in enumerate(submission_data):
        sid = str(sub_item.get("question_id", sub_item.get("id", "")))
        dataset_name = sub_item.get("dataset", "Surgery")
        domain = DATASET_TO_DOMAIN.get(dataset_name, "Surgery")
        
        if domain != "Surgery":
            continue
            
        match_data = testbed_dict[sid] if sid in testbed_dict else (testbed_data[i] if i < len(testbed_data) else None)
        if match_data: 
            surgery_tasks.append({"submission_ref": sub_item, "data": match_data})
            
    total_processed = 0
    
    if surgery_tasks and os.path.exists(DOMAIN_MODELS["Surgery"]):
        model, processor = load_domain_model(DOMAIN_MODELS["Surgery"])
        
        for task in tqdm(surgery_tasks, desc="Surgery 进度"):
            data = task["data"]
            video_path = fix_image_paths(data.get("video_path", []))
            options_str = "\n\nOptions:\n" + "\n".join(data.get("options", [])) if data.get("options") else ""
            
            dataset_name = data.get("dataset", "")
            first_path = video_path[0] if isinstance(video_path, list) and len(video_path) > 0 else ""
            
            if dataset_name == "EgoSurgery" or (dataset_name == "CholecTrack20" and ("VID25" in first_path or "VID111" in first_path)):
                sampling_fps = 1.0
            else:
                sampling_fps = 0.5 
            prompt = """# Role
You are an expert Surgical Video Analyst specializing in egocentric (first-person) medical procedures.

# Rules
When analyzing the frames, you must account for the following complex situations:
1. **Tool Disambiguation:** Distinguish between similar tools (e.g., Graspers vs. Scissors, L-hook vs. Spatula). Focus on the active tips of the instruments.
2. **Visibility & Occlusion:** Tools or anatomical structures may be partially occluded by blood, smoke (from cautery), or folded tissue. Track a tool's last known trajectory if it goes out of view.
3. **Spatial & Hand Tracking:** Differentiate between the main surgeon's instruments (usually entering from bottom/center) and the assistant's (usually entering from sides/top).
4. **Phase Transitions:** Pay attention to micro-actions (e.g., putting down a dissecting tool to pick up a clipping applier) that signal a transition between surgical phases.
5. **Tissue Interaction:** Differentiate between hovering over tissue, retracting it, or actively cutting/coagulating it.

# Workflow
1. Scan for anatomical landmarks, tools, and visual obstructions.
2. Track the continuous movement of instruments across the sampled frames.
3. Cross-reference visual findings with the options.

Please carefully read the question and its options, then select the most appropriate answer. Question: {question_text}{options_str}
The original FPS of the video is {original_fps}. This image set is obtained by sampling at {sampling_fps} fps.
Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). Do not include any other content.

Example response:
{{
    "prediction": "B",
    "reason": "Paris is the capital city of France."
}}
"""
            prompt = prompt.format(
                question_text=data.get("question_text", ""),
                options_str=options_str,
                original_fps=data.get("original_video_fps", 1.0),
                sampling_fps=sampling_fps
            )


            try: 
                response_text = inference_agent(
                    model=model, 
                    processor=processor, 
                    prompt=prompt, 
                    video_path=video_path, 
                    start_frame_index=0, 
                    sampling_fps=sampling_fps  
                )
                task["submission_ref"]["answer"] = parse_answer(response_text)
            except Exception as e: 
                task["submission_ref"]["answer"] = "A"
            total_processed += 1
            
        del model; del processor; gc.collect(); torch.cuda.empty_cache()

    surgery_results = [task["submission_ref"] for task in surgery_tasks]
    with open("surgery.json", "w", encoding="utf-8") as f: 
        json.dump(surgery_results, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    main()