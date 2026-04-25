import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ResearchIntegrityEnv
from env.models import Action, ActionType, SubmitAuditPayload, FlawReport

def main():
    print("Loading PeerGuard LoRA (using standard Transformers for Windows)...")
    model_name = "unsloth/Llama-3-8b-Instruct-bnb-4bit"
    lora_path = "peerguard_lora_final"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base 4-bit model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Apply LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    print("Model loaded successfully!")

    SYS = """You are an FDA Lead Regulator auditing clinical trials.
You will receive a clinical trial methodology section.
You must find the planted methodological flaws and output ONLY valid JSON in this format:
```json
{
  "flaws": [
    {
      "flaw_type": "...",
      "location": "...",
      "description": "..."
    }
  ]
}
```"""

    print("\n--- Evaluating on Task 1 (Unseen Paper) ---")
    env = ResearchIntegrityEnv(seed=9999) # Using an unseen seed
    obs = env.reset("task1_methodology_audit")

    prompt = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Protocol:\n{obs.paper_text}"},
    ]

    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    print("Agent is thinking...\n")
    outputs = model.generate(input_ids=inputs, max_new_tokens=1024, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    
    result = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
    print("--- AGENT AUDIT REPORT ---")
    print(result)

    # Score it
    print("\n--- GRADING ---")
    try:
        t = result.split("```json")[-1].split("```")[0].strip()
        p = json.loads(t)
        
        flaws = [FlawReport(flaw_type=str(f.get("flaw_type","")), location=str(f.get("location","")), description=str(f.get("description",""))) for f in p["flaws"]]
        action = Action(action_type=ActionType.submit_audit, audit_payload=SubmitAuditPayload(flaws=flaws))
        
        _, rw, _, _ = env.step(action)
        print(f"Agent Grader Score: {rw.grader_score:.4f} / 1.0000")
        if rw.grader_score > 0.9:
            print("✅ SUCCESS: The RL Agent caught the methodological flaws perfectly!")
    except Exception as e:
        print(f"Failed to parse or score: {e}")

if __name__ == "__main__":
    main()
