# FF_RAG
Bridging Parameter-Efficient Fine-Tuning and Retrieval-Augmented Generation via Federated Architecture for Deep Integration of Domain Expert and Knowledge
FF-RAG is a novel architecture that integrates Federated Learning (FL), Parameter-Efficient Fine-Tuning (PEFT), and Retrieval-Augmented Generation (RAG) to tackle the challenges of applying large language models (LLMs) to domain-specific tasks while preserving data privacy. This repository provides the implementation of FF-RAG, including the foundational FF-LoRA method designed to integrate global and personalized features, improving model performance in domain-specific applications.

## Overview
As the use of large language models (LLMs) grows, applying them to domain-specific tasks (e.g., healthcare) faces challenges due to:
- 🔒 Data privacy concerns
- 🌐 Data heterogeneity across clients

FF-RAG addresses these challenges by:
1. Integrating **PEFT** to efficiently fine-tune LLMs while preserving general knowledge
2. Combining **RAG** to enhance domain-specific performance through external knowledge retrieval
3. Using **federated learning** to maintain data privacy through decentralized training

The framework enables bidirectional consistency between domain models and retrieval systems, ensuring generated responses align with retrieved documents.

## Key Features

- **FF-LoRA**: Integrates personalized client features with global server-side knowledge to mitigate client drift
- **Dual-task Strategy**: Uses local case bases + global authoritative knowledge base for optimized retrieval
- **Bidirectional Consistency**: Aligns generator outputs with retrieved knowledge sources
- **Privacy Preservation**: Federated learning prevents data leakage by keeping data decentralized
- **Efficient Communication**: Reduced overhead compared to full model fine-tuning

## Architecture

### Core Components
| Component | Function |
|----------|----------|
| **FF-LoRA** | Federated fine-tuning method with two-stage knowledge fusion |
| **Knowledge Boundary** | Combines domain-specific case bases and authoritative knowledge to address hallucinations |
| **Retriever-Generator** | Ensures outputs align with relevant domain-specific information |

## Requirements
pip install -r requirements.txt

## Usage


