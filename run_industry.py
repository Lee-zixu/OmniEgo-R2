import json
import os
import re
import time
import torch
import gc
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

DOMAIN_MODELS = {
    "Industry": "./EgoCross-main/models/industry"
}

LORA_ADAPTERS = {
    # "Industry": "./EgoCross-main/output/egocross_industry",
}

DATASET_TO_DOMAIN = {
    "ENIGMA": "Industry"
}

ENIGMA_TAXONOMY = """
# Strict Vocabulary Constraints (25 Valid Items)
You MUST ONLY identify objects using these precise terms. DO NOT use generic terms like "board" or "button".
- [Measurement Group]: oscilloscope, oscilloscope_probe_tip, oscilloscope_ground_clip
- [Welding Group]: welder_station, welder_base, welder_probe_tip
- [Power Group]: power_supply, power_supply_cables
- [Target Boards]: low_voltage_board, high_voltage_board, low_voltage_board_screen
- [Hand Tools]: pliers, screwdriver, electric_screwdriver, electric_screwdriver_battery
- [Components]: register, battery_connector, socket_1, socket_2, socket_3, socket_4
- [Panel Controls]: left_red_button, left_green_button, right_red_button, right_green_button
"""

COMMON_OUTPUT_REQUIREMENT = """
CRITICAL OUTPUT FORMAT:
First, execute your step-by-step reasoning process inside <thinking> tags. 
ONLY AFTER you have finished thinking, output the final answer in a strict JSON block exactly like this:
```json
{
  "prediction": "A",
  "reason": "Brief summary of the conclusion."
}
Do not include any other text after the JSON block.
"""

SINGLE_AGENT_PROMPTS = {
"Industry_ObjectCounting": ENIGMA_TAXONOMY + """
# Role
You are an elite FPV Industrial Vision Analyst. Your task is to count distinct object types under strict constraints and visual noise.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. [Constraint Override - SUPREME RULE]: Read the question text carefully. If it explicitly states a grouping rule like "(e.g. buttons as a single category)", YOU MUST OBEY THIS. This means all buttons (regardless of color or position) collectively count as exactly ONE item. Do not list or count them separately.
2. [Isolate the Dominant Scene - CRITICAL]: FPV videos often contain random sweeps or blurry transitions. Find the frames where the camera is STABLE, CLEAR, and intentionally focused on a specific target (e.g., staring straight at a control panel). YOU MUST IGNORE frames with motion blur, wide sweeps, or messy background workbenches. They are visual noise.
3. [Option-Guided Filtering]: Check the options. The numbers are very small (e.g., 0, 1, 2, 3). This is a strong hint: DO NOT count background equipment (power_supply, oscilloscope, scattered tools). ONLY count the primary object(s) in the Dominant Scene.
4. [Visual Scratchpad]: Write down your observations strictly applying the rules above. (e.g., "Frames 0-2 are stable and focus on the panel. Frames 3-4 are blurry sweeps, ignoring them. The panel has buttons. Constraint says buttons = 1 category. Total count is 1.")
5. [Final Count]: Output the final integer and select the matching option.
{common_requirement}
""",
    "Industry_NextInteraction": ENIGMA_TAXONOMY + """
# Role
You are an elite Industrial Task Predictor specializing in strict SOP workflows.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. Target Isolation: Which hand/tool does the question ask about?
2. [Vocabulary Mapping]: Note that the options use the verb "contact" (e.g., "contact the battery"). This simply means the operator will "touch", "grab", or "interact with" that object.
3. [Visual Scratchpad]: Look at the final 2-3 frames. What is the target hand doing? What is it moving towards?
4. SOP Logic: Predict the immediate next step. If the hand is reaching towards a panel, it will contact a button. If holding pliers, it will interact with a component or board.
5. Select the option that logically describes the next object to be touched.
{common_requirement}
""",

    "Industry_DominantHeldObject": ENIGMA_TAXONOMY + """
# Role
You are an elite Hand-Object Interaction Analyst focusing on distinguishing highly similar lab tools.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. Target Isolation: Identify the specific hand mentioned. Ignore the other hand completely.
2. Tool Feature Extraction: Look closely at the tip and handle. 
   - Is it a `welder_probe_tip` (used for soldering) or `oscilloscope_probe_tip` (used for testing)? 
   - Is it a manual `screwdriver` or an `electric_screwdriver`?
3. Match your extracted features STRICTLY to the Vocabulary terms.
{common_requirement}
""",

    "Industry_NotVisible": ENIGMA_TAXONOMY + """
# Role
You are an elite Scene Analyst focusing on exhaustive visual scanning to find what is MISSING.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. The Checklist: Treat options A, B, C, and D as your strict search targets. 
2. [Visual Scratchpad]: Search every single frame for these 4 items. Write:
   - "Option A ([Item Name]): Found in Frame [X]."
   - "Option C ([Item Name]): Not found."
3. Strict Elimination: Partial visibility means the object IS visible. Cross off any option you successfully found. Select the ONE option that is completely absent.
{common_requirement}
""",

    "Industry_Spatial": ENIGMA_TAXONOMY + """
# Role
You are an elite FPV Spatial Localization Analyst.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. Dynamic Anchoring: Find the exact frame where the target mentioned in the question is clearest.
2. Estimate the 2D position of the target object. 
3. Relative Positioning: Evaluate its position relative to large anchors (`welder_station`, `power_supply`, `oscilloscope` or the center of the `low_voltage_board`).
4. Select the option that correctly describes this spatial relationship.
{common_requirement}
""",
    "Industry_Temporal": ENIGMA_TAXONOMY + """
# Role
You are a Precise Temporal Video Analyst.

# Source Context
- Question: {question_text}{options_str}
- Total frames: {total_frames}
- CRITICAL: Time = Frame Index * 2.0s.

# Task Instruction (Zero-Shot Chain-of-Thought)
Execute your analysis systematically inside your <thinking> process:
1. Define Micro-Action: What specific physical contact or movement defines the "start" of this action?
2. Boundary Mapping: Find the [Last Inactive Frame] and [First Active Frame].
3. Calculate: Interpolate the exact time in seconds. Select the closest timestamp option.
{common_requirement}
"""
}
def fix_image_paths(paths):
    if isinstance(paths, str):
        return paths.replace("/egocross_testbed/", "./EgoCross-main/datasets/egocross_testbed/")
    return [p.replace("/egocross_testbed/", "./EgoCross-main/datasets/egocross_testbed/") for p in paths]

def load_domain_model(domain):
    model_path = DOMAIN_MODELS[domain]
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa", 
        device_map="cuda:0", 
        trust_remote_code=True
    )
    adapter_path = LORA_ADAPTERS.get(domain)
    
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, min_pixels=50176, max_pixels=360000)
    return model, processor

def parse_answer(response: str):
    prediction = ""
    try: 
        prediction = json.loads(response).get("prediction", "")
    except:
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            try: prediction = json.loads(json_match.group(0)).get("prediction", "")
            except: pass
            
    if not prediction:
        matches = re.findall(r'\b([A-D])\b', response)
        if matches:
            prediction = matches[-1] 
    prediction = str(prediction).strip().upper()
    if prediction not in ["A", "B", "C", "D"]:
        return "A"
    return prediction

def inference_agent(model, processor, prompt, question_type, video_path=None, start_frame_index=0):
    messages = []
    current_max_pixels = 518400 if ("counting" in question_type or "visible" in question_type) else 360000

    if video_path:
        content_list = []
        if isinstance(video_path, list):
            for i, path in enumerate(video_path):
                current_sec = (start_frame_index + i) * 2
                content_list.append({"type": "text", "text": f"<image>{current_sec}s\n"})
                content_list.append({"type": "image", "image": path, "max_pixels": current_max_pixels})

            content_list.append({"type": "text", "text": prompt})
        else:
            content_list.append({"type": "video", "video": video_path, "max_pixels": current_max_pixels})
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
    
    rep_penalty = 1.1
    generated_ids = model.generate(**inputs, max_new_tokens=2048, repetition_penalty=rep_penalty)
    input_length = inputs.input_ids.shape[1]
    response = processor.batch_decode([out_ids[input_length:] for out_ids in generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    del inputs, generated_ids, image_inputs, video_inputs
    torch.cuda.empty_cache()
    
    return response

def process_single_task(model, processor, task_data):
    task_id = task_data["submission_ref"].get("id", "Unknown")
    data = task_data["data"]
    video_path = fix_image_paths(data.get("video_path", [])) 
    
    total_frames = len(video_path) if isinstance(video_path, list) else 1
    options_str = "\n\nOptions:\n" + "\n".join(data.get("options", [])) if data.get("options") else ""
    question_text = data.get('question_text', '')
    question_type = data.get("question_type", "").lower()

    print(f"\n" + "="*60)
    print("="*60)
    
    if "object counting" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_ObjectCounting"]
    elif "next interaction" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_NextInteraction"]
    elif "dominant held-object" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_DominantHeldObject"]
    elif "not visible" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_NotVisible"]
    elif "spatial localization" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_Spatial"]
    elif "temporal localization" in question_type:
        template = SINGLE_AGENT_PROMPTS["Industry_Temporal"]
    else:
        template = SINGLE_AGENT_PROMPTS["Industry_DominantHeldObject"] 

    prompt = template.format(
        question_text=question_text, 
        options_str=options_str,
        total_frames=total_frames,  
        common_requirement=COMMON_OUTPUT_REQUIREMENT
    )

    final_response = inference_agent(model, processor, prompt, question_type, video_path=video_path)
    
    return final_response

def main():
    testbed_path = "datasets/egocross_testbed_imgs.json"
    
    if not os.path.exists(testbed_path):
        return
        
    with open(testbed_path, "r", encoding="utf-8") as f: testbed_data = json.load(f)
    with open("merged_all_answers_ours.json", "r", encoding="utf-8") as f: submission_data = json.load(f)
        
    industry_tasks = []
    testbed_dict = {str(item.get("id", "")).strip(): item for item in testbed_data}
            
    for sub_item in submission_data:
        sid = str(sub_item.get("id", "")).strip()
        dataset_name = sub_item.get("dataset", "")
        if DATASET_TO_DOMAIN.get(dataset_name) == "Industry":
            match_data = testbed_dict.get(sid)
            if match_data: #  and match_data.get("question_type", "") == "object counting"
                industry_tasks.append({"submission_ref": sub_item, "data": match_data})

    model, processor = load_domain_model("Industry")

    for task in tqdm(industry_tasks, desc="Industry"):
        try:
            final_response = process_single_task(model, processor, task)
            task["submission_ref"]["answer"] = parse_answer(final_response)
            task["submission_ref"]["raw_output_final"] = final_response
        except Exception as e:
            task["submission_ref"]["answer"] = "A" 
            
    with open("industry_debug.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
    for item in submission_data:
        item.pop("raw_output_final", None)
            
    with open("industry.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    main()