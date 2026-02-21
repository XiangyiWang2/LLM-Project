# RAG-Optimization-Lab: Vertical QA System with Qwen-SFT and BGE-RAG

This project implements a high-performance, vertical-domain Retrieval-Augmented Generation (RAG) system based on **Qwen-1.5-7B**. By combining **LoRA Fine-tuning (SFT)** for instruction following and **BGE-M3** for semantic retrieval, the system achieves significant performance gains over base LLMs.

Key Features
- **SFT (Supervised Fine-Tuning):** Efficient LoRA adapter training on domain-specific datasets.
- **Advanced RAG:** High-dimensional vector search using FAISS and BGE-M3 embeddings.
- **Ablation Evaluation:** A 3-way evaluation framework comparing Base Model vs. SFT vs. SFT+RAG.
- **Gradio UI:** A professional user interface for real-time interaction and evidence-chain visualization.

Our experiments demonstrate a "Staircase Improvement" in model performance:

| Model Configuration | Exact Match (EM) | F1 Score | Improvement (Rel.) |
| :--- | :---: | :---: | :---: |
| Base Qwen-1.5-7B | 1.50% | 9.48% | Baseline |
| Qwen + LoRA (SFT) | 9.50% | 16.51% | ~6.3x EM |
| Qwen + LoRA + RAG | 22.00% | 31.29% | ~14.6x EM |

The SFT process successfully taught the model the response format, while RAG provided the external "knowledge brain," effectively doubling the performance again.

The model was trained using 4-bit quantization and LoRA. The loss curve shows perfect convergence:
- **Optimizer:** AdamW
- **Learning Rate:** 2e-4 (Linear Decay)
- **Batch Size:** 4 (with Gradient Accumulation)

[Training Overview](report/wandb_training_overview.png)
[Evaluation Report](report/rag_3way_report.png)

 Project Structure
.
├── data/           # FAISS Index & Processed Datasets
├── checkpoints/    # Trained LoRA Adapters
├── rag/            # Retrieval logic & Indexing scripts
├── sft/            # Fine-tuning scripts (PEFT/LoRA)
├── report/         # Experimental Visualizations & Logs
├── ui/             # Gradio Web Interface
└── eval_rag_3way.py # End-to-end Evaluation Pipeline