import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "model"   # folder created after unzip

st.set_page_config(page_title="MedQ&A - Qwen Model", layout="wide")
st.title("🩺 MedQ&A – Your Fine-Tuned Qwen Model")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return tokenizer, model

# Load model once
tokenizer, model = load_model()

def generate_answer(question):
    prompt = (
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("assistant")[-1].strip()

# UI Section
st.markdown("### Ask a medical question")

question = st.text_area("Your question:", height=120)

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = generate_answer(question)
        st.subheader("Answer:")
        st.success(answer)
    else:
        st.warning("Please enter a question.")
