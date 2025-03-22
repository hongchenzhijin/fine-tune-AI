from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import dispatch_model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("final_model")

# Load base model and prepare for offloading
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/DeepSeek-R1-Distill-Llama-8B",
    device_map="auto",  # auto split across GPU and CPU/disk
    offload_folder="offload_folder",  # <-- REQUIRED!
    torch_dtype="auto",
)

# Load adapter with same offload_dir
model = PeftModel.from_pretrained(
    base_model,
    "final_model",
    device_map="auto",
    offload_folder="offload_folder",  # <-- REQUIRED for PEFT too!
)

# Inference
prompt = "What are the causes of skn cancer and how to prevent it?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
