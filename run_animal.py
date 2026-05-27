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
    "Animal": "./EgoCross-main/models/animal"
}

LORA_ADAPTERS = {
    # "Animal": "./EgoCross-main/output/egocross_animal",
}

DATASET_TO_DOMAIN = {
    "EgoPet": "Animal"
}

COMMON_OUTPUT_REQUIREMENT = """
Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). First, output the JSON, then output your reasoning process inside <thinking> tags. Do not include any other content. Even if you cannot find the exact evidence, you MUST CHOOSE one option from A, B, C, or D. Summarize quickly and IMMEDIATELY output the JSON format to avoid truncation.
"""

MULTI_AGENT_PROMPTS = {
    "Role1_Frame_Observer": """# Role
You are an expert Egocentric Vision Frame Observer. You analyze sequential frames extracted from a pet's wearable camera.

# Source Context
- Question to answer: {question_text}
- Possible Options: {options_str}
- Total frames provided: {total_frames} frames.
- The sampling time interval between adjacent frames is 2s.

# Task Instruction
You are provided with exactly {total_frames} frames. Describe EACH frame chronologically. You do not need to output strict JSON. Instead, provide a highly detailed, comprehensive narrative for each frame, clearly labeling the frame number (from 0 to {max_frame_index}) and timestamp. Keep the Question and Options in mind to ensure your descriptions capture relevant clues.

# Focus Areas
1. Subject's Body Parts: Look at the edges of the frame. Are there claws, paw pads, fur, or a snout visible? These may belong to the photographer (the subject animal) itself.
2. Perspective Height: Is the camera looking up at furniture legs (small animal) or looking down at the grass (medium/large animal) or looking from sky (bird)?
3. Target Characteristics: When a target (a human, animal, specific object, or environment mentioned in the question or options) appears, describe its specific visual features in detail (e.g., clothing color, species, fur color, current posture).
4. Spatial Position & Distance: **CRITICAL** For any identified target, explicitly state:
   - Position: Where are they in the frame? (e.g., dead center, top-right corner, entering from the left edge).
   - Distance: How far are they from the camera? (e.g., very close/touching, mid-ground/a few steps away, far background).

# Output Format
Provide a structured text response. Use clear headings for each of the {total_frames} frames.Do not output final answer or your analysis, keep output only has frame description.Do not repeat your output again or exceed total frames number.
Example format:
Frame 0 (0s): The camera is positioned low to the ground. In the dead center, there is a red plastic toy resting on a wooden floor. The environment is an indoor living room. A white furry snout is visible at the very bottom edge.
Frame 1 (2s): The camera has moved forward. The red plastic toy in the center is now much closer and takes up more of the visual field. A human hand (wearing a silver ring) reaches into the frame from the top right, moving towards the toy.
Frame 2 (4s): The toy is no longer visible, and the camera perspective shifts abruptly upwards, showing a plain white wall and the edge of a sofa. No people or animals are currently in the center.
""",

    "Role2_Motion_Analyst": """# Role
You are a precise Video Motion and Animal Interaction Analyst.

# Source Context
- Question to answer: {question_text}
- Possible Options: {options_str}
- Base Frame Descriptions (from Observer): {role1_output}

# Task Instruction
Analyze the CONTINUOUS flow based STRICTLY on the Base Frame Descriptions. Do not invent timestamps, animals, or objects that are not explicitly mentioned in the Observer's notes.
If there is no obvious interactive action, then choose the target that is closest to the animal subject and appears in the frame(Especially in the middle of the frame.), but does not necessarily require a mutual action. 

# Focus Areas (Action & Interaction)
1. Differentiate & Track: Match the target's specific characteristics (from the Observer's notes) across frames to ensure you are tracking the same entity. Paws/snouts at the edges belong to the Subject.
2. Spatial Trajectory & Distance Shift (CRITICAL): Describe how the distance and position change between frames. Is the distance closing (e.g., moving from mid-ground to foreground)? A rapidly closing distance indicates a charge, sprint, or greeting. Is the target moving into the dead center of the frame? 
3. "No Action" Protocol: If the base frames show little to no change, or no targets appear, explicitly state "No significant movement or interaction occurred". DO NOT invent scenarios.
4. Defining "Start" (Temporal): Find the MOMENT OF INITIATION. Interaction starts when intent is shown (e.g., sudden sprint, intense gaze, vocalization, or the moment the target rapidly centers in the FOV), even before physical contact.

# Output Format
Provide a strictly objective, concise summary using this structure:
[Entities Present]: ...
[Movement/Trajectory]: ...
[Interaction Start]: ...
Do not output JSON, internal thoughts, self-corrections, or conversational filler. Do not repeat your output again.
""",

    "Role3_Animal_Identification": """# Role
You are an expert Animal Identification Specialist focusing on egocentric vision.

# Source Context
You are provided with the actual video frames AND the objective reports from your two visual assistants.
- Question: {question_text}{options_str}
- Assistant 1 (Frame-by-Frame Details): {role1_output}
- Assistant 2 (Continuous Motion & Interaction): {role2_output}
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Synthesize your own visual observations of the frames with the textual clues from your assistants to **SELECT the correct species or breed from the given options**.

# Independent Fallback Protocol (CRITICAL)
If the assistants' reports are too brief or fail to identify key elements, **ACT AS A STANDALONE ANALYST**. Directly observe the frames using these rules:
1. Body Part Inspection: Scan frame edges for the Subject's own paws (claws vs pads), snout length, and fur texture.
2. Perspective Height: Analyze camera height. A POV very close to the floor (looking up at furniture) indicates a cat or small dog; a higher POV indicates a medium/large dog.
3. Reflections: Look for mirrors or glass that reveal the Subject's ears or full body.
4. Environment Clues: Match the setting (e.g., cat trees/high shelves for cats vs leashes/grass for dogs) to the Subject.
5. Elimination: Compare clues against options. Eliminate breeds that don't match the observed height or fur color.

# Logical Guidelines
Cross-verify the assistants' claims with the actual visual evidence. If they missed obvious details, trust your own visual analysis.
{common_requirement}Do not repeat your output again.
""",

    "Role3_Animal_Interaction": """# Role
You are an expert Animal Interaction and Object Analyst specializing in egocentric vision.

# Source Context
You are provided with the actual video frames AND the objective reports from your two visual assistants.
- Question: {question_text}{options_str}
- Assistant 1 (Frame-by-Frame Details): {role1_output}
- Assistant 2 (Continuous Motion & Interaction): {role2_output}
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Synthesize your own visual observations of the frames with the textual clues to **IDENTIFY THE TARGET OBJECT OR ENTITY** that the subject is interacting with or moving towards.

# Independent Fallback Protocol (CRITICAL)
If the assistants' reports are too brief or fail to identify key elements, **ACT AS A STANDALONE ANALYST**. Directly observe the frames using these rules:
1. Differentiate Self vs. Other: Paws at edges are the Subject. **Any same-species animal in front of the lens (even at a distance) is the INTERACTIVE TARGET (e.g., "Cat" or "Dog")**.
2. Attribute Matching: Analyze target properties:
    - Biological/Furry: Another animal or human.
    - Liquid/Reflective: Water.
    - Artificial/Shiny/Thin: Plastic or Filament.
3. Engagement Clues: The Target is often in the center of the FOV, being stared at, sniffed, or touched by the Subject's paws/mouth.
4. Depth Scanning: Look beyond the immediate foreground; the Subject may be stalking a target in the mid-field.

# Logical Guidelines
Cross-verify the assistants' claims with the actual visual evidence. If they missed obvious details, trust your own visual analysis.
{common_requirement}Do not repeat your output again.
""",

    "Role3_Animal_Temporal": """# Role
You are a Precise Temporal Video Analyst specializing in animal social behavior.

# Source Context
You are provided with the actual video frames AND the objective reports from your two visual assistants.
- Question: {question_text}{options_str}
- Assistant 1 (Frame-by-Frame Details): {role1_output}
- Assistant 2 (Continuous Motion & Interaction): {role2_output}
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Synthesize your own visual observations with the textual clues to identify the **EARLIEST STARTING POINT** of the specified interaction and select the correct range.

# Independent Fallback Protocol (CRITICAL)
If the assistants' reports are too brief or fail to identify key elements, **ACT AS A STANDALONE ANALYST**. Directly observe the frames using these rules:
1. **Define "Start":** Interaction begins at the **MOMENT OF INITIATION (Intent)**. 
    - If the dog barks (mouth opening/head tossing) or lunges toward a door/person at 26s, the interaction **starts at 26s**, even if the human responds at 28s.
2. **Visual Initiation Signals:** Look for vocalization (mouth movements), intense gaze at a target, or a sudden sprint toward an entity.
3. **Mathematical Mapping:** Video is sampled at {sampling_fps} FPS. (Frame 1=0.0s, Frame 2={time_step:.2f}s, Frame 3={time_step_x2:.2f}s).
4. **Range Strategy:** Each option covers ~xs. Find the **x-second block** that contains the first initiation signal. 
5. **Causal Chain:** If an action at 26s causes a reaction at 28s, use 26s as the start time and match it to the options.

# Logical Guidelines
Cross-verify the assistants' claims with the actual visual evidence. If they missed obvious details, trust your own visual analysis.
{common_requirement}Do not repeat your output again.
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
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True)
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
        match = re.search(r'\b([A-D])\b', response)
        if match:
            prediction = match.group(1)
            
    prediction = str(prediction).strip().upper()
    if prediction not in ["A", "B", "C", "D"]:
        return "A" # 兜底
    return prediction


def inference_agent(model, processor, prompt, video_path=None, is_json_forced=False, start_frame_index=0):
    messages = []
    
    if video_path:
        content_list = []
        if isinstance(video_path, list):
            for i, path in enumerate(video_path):
                current_sec = (start_frame_index + i) * 2
                content_list.append({"type": "text", "text": f"<image>{current_sec}s\n"})
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

def process_single_task_with_agents(model, processor, task_data):
    task_id = task_data["submission_ref"].get("id", "Unknown")
    data = task_data["data"]
    video_path = fix_image_paths(data.get("video_path", [])) 
    
    total_frames = len(video_path) if isinstance(video_path, list) else 1
    max_frame_index = max(0, total_frames - 1)
    
    options_str = "\n\nOptions:\n" + "\n".join(data.get("options", [])) if data.get("options") else ""
    question_text = data.get('question_text', '')
    question_type = data.get("question_type", "").lower()

    
    if total_frames > 10 and isinstance(video_path, list):
        mid = total_frames // 2
        part1_paths = video_path[:mid]
        part2_paths = video_path[mid:]
        prompt_role1_part1 = MULTI_AGENT_PROMPTS["Role1_Frame_Observer"].format(
            question_text=question_text, 
            options_str=options_str,
            total_frames=len(part1_paths),        
            max_frame_index=len(part1_paths) - 1   
        )
        out1 = inference_agent(model, processor, prompt_role1_part1, video_path=part1_paths, is_json_forced=False, start_frame_index=0)
        
        prompt_role1_part2 = MULTI_AGENT_PROMPTS["Role1_Frame_Observer"].format(
            question_text=question_text, 
            options_str=options_str,
            total_frames=len(part2_paths),        
            max_frame_index=len(part2_paths) - 1   
        )
        prompt_role1_part2 += f"\n\n[CRITICAL NOTE] This is the second half of the video. You MUST start labeling from Frame {mid} (Time: {mid*2}s) instead of Frame 0 to maintain continuous chronological order.You need output from Frame {mid} to Frame {max_frame_index}"
        
        out2 = inference_agent(model, processor, prompt_role1_part2, video_path=part2_paths, is_json_forced=False, start_frame_index=mid)
        
        role1_output = out1 + "\n\n" + out2
    else:
        prompt_role1 = MULTI_AGENT_PROMPTS["Role1_Frame_Observer"].format(
            question_text=question_text, 
            options_str=options_str,
            total_frames=total_frames,        
            max_frame_index=max_frame_index   
        )
        role1_output = inference_agent(model, processor, prompt_role1, video_path=video_path, is_json_forced=False, start_frame_index=0)
        
    prompt_role2 = MULTI_AGENT_PROMPTS["Role2_Motion_Analyst"].format(
        question_text=question_text, 
        options_str=options_str, 
        role1_output=role1_output
    )
    role2_output = inference_agent(model, processor, prompt_role2, video_path=video_path, is_json_forced=False)
    print("-" * 60)
    
    if "temporal localization" in question_type:
        template = MULTI_AGENT_PROMPTS["Role3_Animal_Temporal"]
    elif "interaction identification" in question_type:
        template = MULTI_AGENT_PROMPTS["Role3_Animal_Interaction"]
    elif "animal identification" in question_type:
        template = MULTI_AGENT_PROMPTS["Role3_Animal_Identification"]
    else:
        template = MULTI_AGENT_PROMPTS["Role3_Animal_Interaction"]

    if "temporal localization" in question_type:
        prompt_role3 = template.format(
            question_text=question_text, 
            options_str=options_str,
            role1_output=role1_output, 
            role2_output=role2_output,
            total_frames=total_frames,  
            common_requirement=COMMON_OUTPUT_REQUIREMENT,
            sampling_fps=0.5,
            time_step=2.0,
            time_step_x2=4.0
        )
    else:
        prompt_role3 = template.format(
            question_text=question_text, 
            options_str=options_str,
            role1_output=role1_output, 
            role2_output=role2_output,
            total_frames=total_frames,  
            common_requirement=COMMON_OUTPUT_REQUIREMENT,
        )
    final_response = inference_agent(model, processor, prompt_role3, video_path=video_path, is_json_forced=False, start_frame_index=0)
    print("="*60)
    
    return final_response, role1_output, role2_output

def main():
    testbed_path = "datasets/egocross_testbed_imgs.json"
    
    if not os.path.exists(testbed_path):
        return
        
    with open(testbed_path, "r", encoding="utf-8") as f: testbed_data = json.load(f)
    with open("submission_template.json", "r", encoding="utf-8") as f: submission_data = json.load(f)
        
    animal_tasks = []
    testbed_dict = {str(item.get("id", "")).strip(): item for item in testbed_data}
            
    for sub_item in submission_data:
        sid = str(sub_item.get("id", "")).strip()
        dataset_name = sub_item.get("dataset", "")
        if DATASET_TO_DOMAIN.get(dataset_name) == "Animal":
            match_data = testbed_dict.get(sid)
            if match_data:
                animal_tasks.append({"submission_ref": sub_item, "data": match_data})
            
    if not animal_tasks:
        return

    model, processor = load_domain_model("Animal")
    for task in tqdm(animal_tasks, desc="Animal", position=0, leave=True):
        try:
            final_response, role1_out, role2_out = process_single_task_with_agents(model, processor, task)
            
            task["submission_ref"]["answer"] = parse_answer(final_response)
            task["submission_ref"]["raw_output_role1"] = role1_out
            task["submission_ref"]["raw_output_role2"] = role2_out
            task["submission_ref"]["raw_output_final"] = final_response
            
        except Exception as e:
            task["submission_ref"]["answer"] = "A" 
            
    with open("submission_debug.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
    for item in submission_data:
        item.pop("raw_output_role1", None)
        item.pop("raw_output_role2", None)
        item.pop("raw_output_final", None)
            
    with open("submission.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    main()