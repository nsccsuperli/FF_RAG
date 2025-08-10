# FF-RAG
**Bridging Domain Experts and Domain Knowledge via a Federated Learning Framework for Controlled Model Personalization**

FF-RAG is a modular framework that combines **Federated Learning (FL)**, **Parameter-Efficient Fine-Tuning (PEFT)**, and **Retrieval-Augmented Generation (RAG)** to deploy large language models (LLMs) on domain-specific tasks **with privacy preservation and controllable personalization**.  
This repository includes the core implementation, featuring **FF-LoRA**, a two-stage fusion method that blends global and client-specific features to mitigate client drift and improve domain performance.

---

## Overview
Applying general-purpose LLMs to domains such as healthcare is challenging due to:
- 🔒 **Privacy constraints** on sensitive data
- 🌐 **Client heterogeneity** (non-IID data, resource disparity)
- 🧠 **Domain hallucinations** from models lacking grounded domain knowledge
- 🔁 **Retriever–Generator misalignment**, causing off-policy or inconsistent outputs

**FF-RAG** addresses these by:
- 🔄 **FF-LoRA**: Integrates client personalization with global knowledge in an FL loop, reducing client drift and preserving local preferences.
- 🧩 **Dual-Boundary Knowledge**: Constructs **authoritative** and **case-based** knowledge sources to bound generation and curb hallucinations.
- 🎯 **Retriever–Generator Agreement**: Encourages outputs that stay consistent with retrieved, domain-relevant evidence.

---

## ✨ Features
- **Two-Stage Knowledge Fusion (FF-LoRA)**: Stackable adapters for global + client features.
- **Dual Knowledge Bases**: Authoritative definitions + case dialogs/records to ground reasoning.
- **Privacy-Preserving Training**: FL enables on-prem training without centralizing raw data.
- **Pluggable RAG**: Works with your existing retriever/index and domain corpora.

---

## 🚀 Quick Start
The framework enables bidirectional consistency between domain models and retrieval systems, ensuring generated responses align with retrieved documents.

### Core Components
| Component | Function |
|----------|----------|
| **FF-LoRA** | Federated fine-tuning method with two-stage knowledge fusion |
| **Knowledge Boundary** | Combines domain-specific case bases and authoritative knowledge to address hallucinations |
| **Retriever-Generator** | Ensures outputs align with relevant domain-specific information |

## FF-LoRA

### Requirements
pip install -r requirements.txt

### Data
* In the general The training dataset of WizardLLM has already been downloaded and split in ./data_wiz/ fold.

## Running the experiments
* To run the FF-LoRA algorithm (--stacking: True)  in a heterogeneous LoRA setting:

```bash
python main.py \
    --global_model 'skyline2006/llama-7b' \
    --data_path "./data_wiz" \
    --output_dir './FF-LoRA-llama7b-wiz-heter/' \
    --num_communication_rounds 3 \
    --local_num_epochs 1 \
    --stacking True \
    --heter True
```    

* To run the FF-LoRA algorithm (--stacking: True) in a homogeneous LoRA setting:
```bash
python main.py \
    --global_model 'skyline2006/llama-7b' \
    --data_path "./data_wiz" \
    --output_dir './FF-LoRA-llama7b-wiz-homo/' \
    --num_communication_rounds 3 \
    --local_num_epochs 1 \
    --stacking True 
``` 
* To evaluate on LLM harness, try:
```
lm_eval --model_args pretrained=./FF-LoRA-llama7b-wiz-homo/,parallelize=True,load_in_4bit=False, --tasks mmlu --num_fewshot 5 --batch_size 16 --output_path ../FF-LoRA-llama7b-wiz-homo/
```
* To evaluate on MT-Bench, please follow the instructions on their websites: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
-----

## Continue
We have shown a part of the core code. Because we applied the algorithm to the medical industry, some data and tasks were mixed. We will complete the update after desensitization.

## 🙏 Acknowledge
- [PEFT](https://github.com/huggingface/peft) - FLoRA 


