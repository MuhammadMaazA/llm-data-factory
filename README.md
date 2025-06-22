# 🤖 llm-data-factory: A Synthetic Data Weaver

This project demonstrates how to use a large, powerful "Teacher" language model (like GPT-4) to generate a high-quality synthetic dataset, which is then used to fine-tune a much smaller, more efficient "Student" language model (like Microsoft's Phi-3-mini).

The goal is to create a specialized, cost-effective classifier for a real-world task—in this case, classifying customer support tickets—without needing a large, hand-labeled dataset. This approach showcases modern MLOps techniques like synthetic data generation and model distillation.

**Live Demo App:** `[Link to your deployed Streamlit/Gradio app - e.g., on Hugging Face Spaces]`

---

## 🚀 Core Idea & Motivation

In many real-world machine learning projects, a major bottleneck is the lack of large, high-quality labeled datasets. This project tackles that problem head-on.

1.  **Problem:** We need an accurate model to classify customer support tickets, but we only have a small number of labeled examples.
2.  **Hypothesis:** We can use a powerful, general-purpose LLM (a "Teacher") to understand the task from just a few examples and generate thousands of new, realistic examples.
3.  **Solution:** We then use this rich, synthetic dataset to fine-tune a small, open-weight LLM (a "Student").
4.  **Result:** The final "Student" model is highly specialized, fast, cheap to run, and can be deployed anywhere, all while achieving performance comparable to models many times its size.

This project covers the full AI lifecycle: **Data Scarcity → Data Generation → Efficient Fine-Tuning → Evaluation → Deployment.**

![Workflow Diagram](https://i.imgur.com/uTjZg93.png) _A high-level overview of the project pipeline._

---

## 🛠️ Tech Stack

* **Teacher Model (Data Generation):** OpenAI GPT-4
* **Student Model (Fine-Tuning):** `microsoft/phi-3-mini-4k-instruct`
* **Frameworks:** PyTorch, Hugging Face `transformers`, `datasets`
* **Fine-Tuning:** `peft` (for QLoRA), `trl` (SFTTrainer), `bitsandbytes`
* **Data Handling:** Pandas, JSON
* **Demo App:** Streamlit / Gradio
* **Evaluation:** Scikit-learn

---

## 📁 Repository Structure

llm-data-factory/
├── .gitignore
├── README.md              # You are here!
├── requirements.txt         # Project dependencies
|
├── data/
│   ├── seed_examples.json        # ~15-20 high-quality examples to guide the Teacher model
│   ├── synthetic_data.json       # The final 1000+ example dataset (Generated)
│   └── test_data.json            # A held-out test set from a real dataset for evaluation
|
├── scripts/
│   ├── 01_generate_synthetic_data.py # Script to call the Teacher API and generate data
│   └── 02_finetune_student_model.py  # Main training script for the Student model
|
├── app/
│   ├── app.py                      # The Streamlit/Gradio demo app
│   └── inference.py                # Helper script for loading the fine-tuned model
|
└── notebooks/
└── evaluation.ipynb            # Jupyter Notebook for model evaluation and creating reports


---

## ⚙️ How to Run This Project

Follow these steps to replicate the project on your own machine.

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone [https://github.com/MuhammadMaazA/llm-data-factory.git](https://github.com/MuhammadMaazA/llm-data-factory.git)
cd llm-data-factory
pip install -r requirements.txt
2. Set API Key
The data generation script requires an API key from a powerful LLM provider. Set it as an environment variable.

Bash

export OPENAI_API_KEY='your-openai-api-key'
3. Generate the Synthetic Data
Run the generation script. This will use the examples in data/seed_examples.json to prompt the Teacher model and create data/synthetic_data.json.

Bash

python scripts/01_generate_synthetic_data.py
This might take some time and incur API costs, depending on the number of samples you generate.

4. Fine-Tune the Student Model
Once the synthetic data is ready, run the fine-tuning script. This will train the Phi-3-mini model using QLoRA for efficiency.

Bash

python scripts/02_finetune_student_model.py
The final model artifacts will be saved to the ./final_student_model directory.

5. Evaluate and Launch the Demo
Open the notebooks/evaluation.ipynb notebook to run the final evaluation on the test_data.json and see the performance metrics.
Launch the interactive demo app:
<!-- end list -->

Bash

streamlit run app/app.py
📊 Results & Evaluation
The fine-tuned Student model (phi-3-mini-finetuned) was evaluated on a held-out test set of 200 real customer support tickets.

Classification Report [Placeholder for your results]
Precision	Recall	F1-Score	Support
Urgent Bug	0.92	0.90	0.91	50
Feature Request	0.95	0.96	0.95	70
How-To Question	0.94	0.95	0.94	80
Accuracy			0.94	200

Export to Sheets
Model Performance Comparison [Placeholder for your results]
Model	Accuracy	Cost per 1M Tokens	Size
gpt-4-turbo (Teacher)	97.5%	$10.00	~1.7T
phi-3-mini-base (Untrained Student)	62.0%	~$0.25	3.8B
phi-3-mini-finetuned (Our Model)	94.0%	~$0.25	3.8B

Export to Sheets
As shown, our fine-tuned student model achieves performance remarkably close to the powerful Teacher model but at a fraction of the computational cost, proving the effectiveness of this approach.

✨ Demo
[Placeholder for a screenshot or GIF of your Streamlit app in action]

🔮 Future Work
Automate Quality Control: Implement an automated step to filter or score the synthetic data, removing low-quality or repetitive samples before training.
Experiment with Student Models: Swap out Phi-3-mini for other small models like Gemma 2B or Qwen 1.5B to compare performance.
Expand Label Taxonomy: Increase the number of classification labels to handle more nuanced support ticket types.