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
    "XSports": "./EgoCross-main/models/xsports"
}

LORA_ADAPTERS = {
    # "XSports": "./EgoCross-main/output/egocross_xsports",
}

DATASET_TO_DOMAIN = {
    "ExtrameSportFPV": "XSports"
}

COMMON_OUTPUT_REQUIREMENT = """
Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). First, output the JSON, then output your reasoning process inside <thinking> tags. Do not include any other content. Even if you cannot find the exact evidence, you MUST CHOOSE one option from A, B, C, or D. Summarize quickly and IMMEDIATELY output the JSON format to avoid truncation.
"""

MULTI_AGENT_PROMPTS = {
    "Role1_Frame_Observer": """# Role
You are a professional FPV extreme sports identification expert. You are keenly aware that the first-person view (FPV) camera is **fixed to the participant's head or chest**.

# Source Context
- Question to answer: {question_text}
- Possible Options: {options_str}
- Total frames provided: {total_frames} frames.
- The sampling time interval between adjacent frames is 2 seconds.

# Task Instruction
Please do not provide a frame-by-frame narrative description. Your core task is to quickly scan all video frames and, by identifying the environmental scene and the equipment/limb features at the edges of the frame, **precisely determine which of the following 7 extreme sports is currently being performed**:

# Sport Feature Matching Guide (CRITICAL)
1. Dirt Bike / MTB: The subject usually wears a helmet or other protective gear visible at the bottom of the frame, the bike's front/handlebars typically appear at the bottom center, and the scene is dirt, mud, or mountainous.
2. Snow Bike: Also wears a helmet/pads and has bike handlebars visible, but the distinguishing difference is that the surrounding scene is entirely **snow**.
3. Skiing: The scene is on snow, and **two skis** can be seen at the bottom of the frame (if backlit, the shadows of the person and skis might appear in front).
4. Paragliding: Legs and feet are usually raised and off the ground, the camera sways noticeably, and paraglider lines or canopy parts can be seen at the bottom or on the left/right sides of the frame.
5. Speedriding / Snow Paragliding: The subject also wears two skis on their feet, but unlike regular skiing, the subject is usually **at a distance above the ground (in an airborne state)**, and paraglider components appear on the left and right sides of the frame.
6. Jetski: The scene is entirely on **water**, and the front/handlebars of the jetski appear at the bottom of the frame.
7. Parkour: The subject stands on the ground without the use of any vehicles. Both arms can occasionally be seen swinging at the edges of the frame, and the scene is generally urban streets or buildings/rooftops.

# Output Format
Please provide a concise, structured text response:
[Identified Extreme Sport]: (Insert one of the 7 sports listed above)
[Visual Feature Evidence]: (Briefly describe the environment and equipment features to support your judgment. **STRICT LIMIT: This description must absolutely NOT exceed 4 sentences!**)

Note: Do not output JSON, and do not answer the final question here.
""",

    "Role2_Motion_Analyst": """# Role
You are an expert FPV Extreme Sports Trick Theory and Visual Prediction Specialist.

# Source Context
- Question: {question_text}
- Options: {options_str}
- Identified Sport (from Assistant 1): {role1_output}

# Task Instruction
**ABSOLUTELY DO NOT analyze the actual video frames.** Your task is to focus solely on the question, the options, and the sport identified by Assistant 1. You must identify the "trick" or "trick sequence" mentioned in the question or options, and provide professional explanations and expected FPV visual changes based on the characteristics of that specific extreme sport.

# Deduction Logic & Output Format (CRITICAL)
Evaluate the content characteristics of the options and strictly choose ONE of the following **two cases** for your formatted output:

**Case 1: If the options DO NOT contain trick actions or trick sequences** (e.g., the options are just timestamps like 12.5s, 2.3s, etc.), this means the trick action is only present in the question text (e.g., "When does the 'walk' action begin?").
Please output the following two items:
[What the trick "xxx" might mean in this sport]: (Replace "xxx" with the specific trick from the question. Explain its meaning in a few sentences. E.g., In parkour, "walk" usually means tightrope walking or balancing on a narrow edge).
[Expected visual cues for the trick "xxx" in the video frames for this sport]: (Describe what specific camera changes or features are expected in the FPV perspective when this action is performed).

**Case 2: If the options DO contain trick actions or trick sequences** (e.g., A: Vault -> Run -> Walk, B: Curve right).
Please output the following two items:
[What the tricks/sequences in options A, B, C, and D might respectively mean in this sport]: (List A, B, C, and D separately, providing a few sentences explaining the meaning of each action or sequence).
[Expected visual cues for the tricks/sequences in options A, B, C, and D in the video frames for this sport]: (List A, B, C, and D separately, detailing what specific camera changes or features are expected in the FPV perspective for each action or sequence).

Note: Do not output any conversational filler. Directly output the corresponding two bracketed [] tags based on the current question's case.
""",

    "Role3_XSports_SportID": """# Role
You are an expert Sport Identification Referee. The camera is strictly fixed to the participant's head or chest.

# Source Context
- Question: {question_text}{options_str}
- Asst 1 (Sport/Environment): {role1_output}
- Asst 2 (Visual Cues): {role2_output}
- Total frames: {total_frames} 
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Your goal is to identify the correct extreme sport. Follow these exact steps:
1. Synthesize Context: Read Assistant 1's sport identification and Assistant 2's theoretical cues.
2. Direct Verification: Observe the actual video frames to verify if the physical environment and visible FPV gear match Assistant 1's conclusion.
3. Apply PROCESS OF ELIMINATION: Evaluate options A, B, C, and D. Discard any option that contradicts the visual evidence of the gear/environment, and select the final correct answer.

# Independent Fallback Protocol (CRITICAL)
Activate ONLY IF assistants are erroneous, vague, or contradictory. Observe frames using these rules:
1. Environment Scan: Check the background. Is it entirely water, snow-covered mountains, urban streets, or air?
2. Gear/Body Scan: Look at the bottom edge of the frame. What is the camera mounted on? (e.g., bare hands, handlebars, ski tips).
3. Meta-Formula Matching: Combine the environment and gear to deduce the sport.
   - Water + Handlebars = Jetski
   - Snow + Two Parallel Tips = Skiing
   - Urban/Street + Bare Hands/No Gear = Parkour

# Logical Guidelines
The FPV gear is the ground truth. If gear is present, it defines the sport regardless of background complexity.
{common_requirement}
""",

    "Role3_XSports_SpecialAction": """# Role
You are an expert FPV Special Action Referee. The camera is fixed to the athlete's head or chest.

# Source Context
- Question: {question_text}{options_str}
- Asst 1 (Sport): {role1_output}
- Asst 2 (Trick Cues): {role2_output}
- Total frames: {total_frames} (Interval: 2s)
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Your goal is to identify the specific trick or maneuver being performed. Follow these exact steps:
1. Extract Checklist: Review Assistant 2's expected visual cues for the tricks listed in the options.
2. Visual Search: Observe the video frames, focusing on the most intense moments, to find which expected camera dynamics or limb movements actually occur.
3. Apply PROCESS OF ELIMINATION: Evaluate options A, B, C, and D. Discard any option whose required visual cue is missing from the footage, and select the final correct answer.
4. Hand-Support Veto: For Parkour, "Jump" means pure spatial shift with NO hands visible. "Vault" REQUIRES a hand or arm to appear in the lower frame to support the body. Veto "Vault" if no hands are seen.
5. Continuous vs. Abrupt Veto: "Spin/Roll" requires massive horizon rotation. "Climb" requires slow vertical shifts facing a wall.
6. Perspective Shift: Check if the camera purely drops/moves forward (legs only) or twists around an object.

# Independent Fallback Protocol (CRITICAL)
Activate if assistants are erroneous, vague, or contradictory. Observe frames using these rules:
1. Perspective Physics: "Jump" = pure spatial shift with NO hands. "Vault" = athlete's own hands must appear to push off.
2. Rotation: "Spin/Roll" = 360-degree horizon rotation relative to the head-mounted camera.

{common_requirement}
""",

    "Role3_XSports_ActionSequence": """# Role
You are an expert FPV Action Sequence Analyst. The camera is fixed to the athlete's head or chest.

# Source Context
- Question: {question_text}{options_str}
- Asst 1 (Sport): {role1_output}
- Asst 2 (Sequence Cues): {role2_output}
- Total frames: {total_frames} (Interval: 2s)
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Your goal is to identify the correct chronological sequence of movements. Follow these exact steps:
1. Understand Expected Sequences: Review Assistant 2's breakdown of the expected visual cues for each sequence option.
2. Map the Timeline: Watch the video from start to finish, breaking it into distinct chronological stages (e.g., Early, Middle, Late frames) based on changing visual features or displacement sizes.
3. Apply PROCESS OF ELIMINATION: Compare your mapped timeline against options A, B, C, and D. Discard any option whose sequence order contradicts your visual timeline, and select the correct answer.

# Independent Fallback Protocol (CRITICAL)
Activate if assistants are erroneous, vague, or contradictory. Observe frames using these rules:
1. Stage Breakdown: Divide video into chronological stages.
2. Movement Logic: Distinguish "Run" vs "Walk" by the **magnitude of spatial displacement** between adjacent frames. Run = Large background shift; Walk = Small background shift.
3. Sequential Elimination: Veto options if any single stage contradicts the visual displacement or perspective shift.

# Logical Guidelines
Sequential veto: If the displacement size in any stage (e.g., Run) doesn't match the speed required by the option, eliminate the whole sequence.
{common_requirement}
""",

    "Role3_XSports_NextDirection": """# Role
You are an expert FPV Trajectory Predictor. The camera is fixed to the athlete's head or chest.

# Source Context
- Question: {question_text}{options_str}
- Asst 1 (Sport): {role1_output}
- Asst 2 (Direction Cues): {role2_output}
- Total frames: {total_frames} (Interval: 2s)
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Your goal is to predict the athlete's next intended direction. Follow these exact steps:
1. Understand Trajectory Cues: Review Assistant 2's visual criteria for differentiating trajectories (e.g., Curve vs. Turn).
2. Isolate Final Frames: Focus your visual observation strictly on the last 2 to 4 seconds of the video frames to detect sudden mutations in camera tilt or orientation.
3. Apply PROCESS OF ELIMINATION: Evaluate options A, B, C, and D. Discard any option that does not match the final camera tilt/orientation, and select the correct answer.

# Independent Fallback Protocol (CRITICAL)
Activate if assistants are erroneous, vague, or contradictory. Observe frames using these rules:
1. The "Last-Frame Mutation" Law: Ignore the early trajectory. Look ONLY at the last 2 to 4 seconds. 
2. Detect Tilt: If the camera/gear was straight but suddenly tilts sharply in the final frames, that tilt is the intended next direction.
3. Nuance Check: Differentiate a "Curve" (long, smooth, sustained banking angle) from a "Turn/Right/Left" (direct, abrupt steering vector).
4. S-Shape Rule: "Left then right" requires a visible S-shape transition in the wake or trajectory.

# Logical Guidelines
In FPV, where the head looks, the body follows. Use the sudden camera shift in the final 2 seconds to eliminate wrong options.
{common_requirement}
""",

    "Role3_XSports_Temporal": """# Role
You are an expert Precise Temporal Analyst. The camera is fixed to the athlete's head or chest.

# Source Context
- Question: {question_text}{options_str}
- Asst 1 (Sport): {role1_output}
- Asst 2 (Trick Cues): {role2_output}
- Total frames: {total_frames}
- The video consists of {total_frames} frames, with a frame interval of 2s.

# Task Instruction
This is a **MULTIPLE-CHOICE SELECTION** task. Your goal is to identify the earliest starting timestamp of the specified action. Follow these exact steps:
1. Define the Action: Read Assistant 2's explanation to understand what the target action visually looks like.
2. Locate the Boundaries: Scan the video frames to find the exact moment the visual cue transitions from intent/approach to physical execution.
3. Apply PROCESS OF ELIMINATION: Map your findings to the 2.0-second frame intervals. Evaluate the timestamp options A, B, C, and D. Discard any timestamp that is clearly too early or too late, and select the closest valid option.

# Independent Fallback Protocol (CRITICAL)
Activate if assistants are erroneous, vague, or contradictory. Observe frames using these rules:
1. State Bounding: Find the [Last Pre-Action Frame] (obstacle is approaching but no interaction yet) and the [First Action Frame] (interaction/airborne state is undeniably happening).
2. Upper Bound Elimination: The action MUST start before the [First Action Frame]. If fully executing at 14s, eliminate options >= 14s.
3. Mathematical Mapping: Video is sampled at {sampling_fps} FPS. (Frame 1=0.0s, Frame 2={time_step:.2f}s, Frame 3={time_step_x2:.2f}s).
4. Closest Match: From the remaining valid options, select the timestamp mathematically closest to your deduced bounding interval.

# Logical Guidelines
The 2s interval is the mathematical basis. The "Start" is the transition frame. Eliminate all options outside this specific 2-second window transition.
{common_requirement}
"""
}


GLOBAL_SYSTEM_PROMPT = """You are an Elite FPV (First-Person View) Extreme Sports AI Analyst, specializing in embodied visual cognition, spatio-temporal physics, and biomechanics. 

Your core objective is to analyze sequences of head/chest-mounted camera frames to perform precise sport identification, temporal action localization, and trajectory prediction.

CRITICAL FPV AXIOMS (Apply these to all reasoning):
1. Embodied Camera Physics: The camera IS the athlete's body. A sudden horizon tilt means a body roll/turn. Vertical camera drop means a jump or fall. Rate of background displacement defines speed.
2. Peripheral Ground Truth: The bottom 20% and extreme edges of the frame are your absolute truth for gear identification. Bare hands = Parkour; Parallel tips = Skiing/Speedriding; Handlebars = Bike/Jetski. 
3. Spatio-Temporal Consistency: Video is sampled at discrete intervals (e.g., 2s). You must calculate the logical gap between frames. If an obstacle is far in Frame A and passed in Frame B, the action occurred in that specific interval.
4. Process of Elimination: Never guess based on overall aesthetics. Ruthlessly veto options that violate visual physics (e.g., veto "Vault" if no hands make contact; veto "Spin" if the horizon remains static).

Maintain absolute analytical rigor. Base all deductions strictly on visible morphological features, environmental context, and sequential state changes. Do not hallucinate actions that lack direct visual evidence."""


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
        return "A" 
    return prediction

def inference_agent(model, processor, prompt, video_path=None, is_json_forced=False, start_frame_index=0):
    messages = [
            {"role": "system", "content": [{"type": "text", "text": GLOBAL_SYSTEM_PROMPT}]}
        ]
    
    if video_path:
        content_list = []
        if isinstance(video_path, list):
            for i, path in enumerate(video_path):
                content_list.append({"type": "image", "image": path, "max_pixels": 360000})
                current_sec = (start_frame_index + i) * 2
                content_list.append({"type": "text", "text": f"<image>{current_sec}s\n"})
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
    """三阶段多智能体管线执行"""
    task_id = task_data["submission_ref"].get("id", "Unknown")
    data = task_data["data"]
    video_path = fix_image_paths(data.get("video_path", [])) 
    
    total_frames = len(video_path) if isinstance(video_path, list) else 1
    max_frame_index = max(0, total_frames - 1)
    
    options_str = "\n\nOptions:\n" + "\n".join(data.get("options", [])) if data.get("options") else ""
    question_text = data.get('question_text', '')
    question_type = data.get("question_type", "").lower()

    print(f"\n" + "="*60)
    print("="*60)
    
    
    prompt_role1 = MULTI_AGENT_PROMPTS["Role1_Frame_Observer"].format(
        question_text=question_text, 
        options_str=options_str,
        total_frames=total_frames,        
        max_frame_index=max_frame_index   
    )
    role1_output = inference_agent(model, processor, prompt_role1, video_path=video_path, is_json_forced=False, start_frame_index=0)
        
    print("-" * 60)
    
    prompt_role2 = MULTI_AGENT_PROMPTS["Role2_Motion_Analyst"].format(
        question_text=question_text, 
        options_str=options_str, 
        role1_output=role1_output
    )
    role2_output = inference_agent(model, processor, prompt_role2, video_path=None, is_json_forced=False)

    print("-" * 60)
    
    question_type_lower = question_type.lower()
    
    if "temporal localization" in question_type_lower:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_Temporal"]
    elif "next direction" in question_type_lower:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_NextDirection"]
    elif "action sequence" in question_type_lower:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_ActionSequence"]
    elif "special action" in question_type_lower:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_SpecialAction"]
    elif "sport identification" in question_type_lower:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_SportID"]
    else:
        template = MULTI_AGENT_PROMPTS["Role3_XSports_SpecialAction"] # 兜底

    if "temporal localization" in question_type_lower:
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
        
    xsports_tasks = []
    testbed_dict = {str(item.get("id", "")).strip(): item for item in testbed_data}
            
    for sub_item in submission_data:
        sid = str(sub_item.get("id", "")).strip()
        dataset_name = sub_item.get("dataset", "")
        if DATASET_TO_DOMAIN.get(dataset_name) == "XSports":
            match_data = testbed_dict.get(sid)
            if match_data:
                xsports_tasks.append({"submission_ref": sub_item, "data": match_data})
            
    if not xsports_tasks:
        return

    model, processor = load_domain_model("XSports")
    for task in tqdm(xsports_tasks, desc="XSports", position=0, leave=True):
        try:
            final_response, role1_out, role2_out = process_single_task_with_agents(model, processor, task)
            
            task["submission_ref"]["answer"] = parse_answer(final_response)
            task["submission_ref"]["raw_output_role1"] = role1_out
            task["submission_ref"]["raw_output_role2"] = role2_out
            task["submission_ref"]["raw_output_final"] = final_response
            
        except Exception as e:
            task["submission_ref"]["answer"] = "A" 
            
    with open("xsports_debug.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
    for item in submission_data:
        item.pop("raw_output_role1", None)
        item.pop("raw_output_role2", None)
        item.pop("raw_output_final", None)
            
    with open("xsports.json", "w", encoding="utf-8") as f: 
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    main()