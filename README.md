# 💡 GenAI FastTrack Portfolio Project

Welcome to **GenAI FastTrack**, a portfolio-ready showcase of Generative AI applications across both vision (DCGAN) and reasoning systems (LLM-enhanced fraud detection).

This project was inspired by a conversation I had during a recent industry forum with a member of a **Fraud Detection & Risk Analytics team** in Shinhan Bank. They emphasized that the **future of auditing, compliance, and financial transaction monitoring** lies not just in traditional models — but in explainable AI systems that can **track user behavior, detect anomalies, and assist analysts with decision-ready insights**.

As someone deeply interested in the **application of Data Science in FinTech**, I built this project to explore how **hybrid systems — combining rule-based engines, LLMs, and visualization tools — can support real-world tasks** such as:

- 🚨 Identifying potentially fraudulent transactions
- 🧾 Explaining why an action is flagged, for audit compliance
- 📊 Monitoring transaction score trends and histories

---

Rather than focusing solely on algorithmic accuracy, this system is designed to be **interpretable, interactive, and actionable** — reflecting the kind of tooling actual **risk and fraud analysts at companies like American Express or PayPal** rely on. With this in mind, I aimed to build my knowledge in the application of GenAI in both **visual(DCGAN) and textual(LLM) manner** with the following GenAI agents with some **interactive features from the scratch** to pursue my interest.

### ✨ Future directions I plan to explore:

- Integrating this engine with **FastAPI + AWS Lambda** for real-time scoring APIs
- Enhancing the scoring logic using **retrieval-augmented generation (RAG)** with domain-specific knowledge
- Deploying a dashboard for **batch processing + alerting** of suspicious clusters in transactions

---

This portfolio not only showcases the **technical proficiency** — from GANs to transformers to Streamlit — but also reflects my desire to **solve meaningful, domain-specific problems** that FinTech firms are actively investing in today.

> 🎯 **Why this matters**: In modern FinTech workflows, the ability to generate, interpret, and act on fraud signals is crucial. This project bridges technical implementation with real decision science use cases in the field.


## 📁 Project Structure


```bash
genai_fasttrack/
├── assets/
│   ├── default_page_UI.png
│   ├── detailed_explanation.png
│   ├── evaluation.png
│   ├── show_fraud_history.png
│   ├── show_transaction_history.png
├── data/
│   └── MNIST/
│       └── raw/
├── genai_rl_agents/
│   ├── __pycache__/
│   ├── dcgan_mnist.py
│   ├── fraud_risk_agent.py
│   ├── fraud_risk_streamlit.py
│   ├── fraud_rules.json
│   └── output/
│       ├── fake_samples_epoch0_step200.png
│       ├── fake_samples_epoch0_step400.png
│       ├── fake_samples_epoch13_step400.png
├── transformers_llm/
│   ├── huggingface_inference.py
│   └── miniGPT_sample.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🧠 1. DCGAN for MNIST (`dcgan_mnist.py`)

Train a Deep Convolutional GAN to generate handwritten digits from the MNIST dataset.

- Generator and Discriminator implemented from scratch
- Custom training loop with `torchvision.utils.make_grid`
- Model output saved under `output/` after each epoch

### 📸 Sample Output (Epoch 13)

![Generated Digits](genai_rl_agents/output/fake_samples_epoch13_step400.png)
> *Digits generated after 13 epochs of DCGAN training using PyTorch*

> *Limited number of training samples considering current condition of military base with limited computing resource*

---

## 🛡️ 2. Fraud Risk Agent (`fraud_risk_agent.py`)

A hybrid risk engine that classifies transactions using:

- ✅ Rule-based scoring from `fraud_rules.json`
- 🤖 Optional LLM-based enhancement
- 🔍 Returns score, risk level, reasons, and recommendations

Designed for use in batch scoring, dashboards, or REST API backends.

---

## 🌐 3. Streamlit UI (`fraud_risk_streamlit.py`)

Interactive web interface for entering or uploading transaction data and reviewing fraud assessments in real time. Built with **Streamlit**, this front-end empowers real-time exploration of transaction risk factors.

### 🖼️ Home Interface

This is the default landing page for users to begin manual input or upload a transaction file.

![Streamlit Home](assets/default_page_UI.png)

---

### 🔍 Risk Breakdown for Sample Case

> 💳 “$3200 online purchase made in Japan at 2 AM using a new iPhone by a user from Singapore with only 12 days of account age.”

Once evaluated, the app displays a summary of:
- Fraud Score (`0–100`)
- Risk Level (Low, Medium, High)
- Action Recommendation (Approve, Investigate, Block)
- Interpretable Reasons

![Streamlit Result](assets/evaluation.png)

---

### 📊 Visualisation on Fraud Score History

The app plots a real-time graph of fraud scores from past evaluations for comparison and trend monitoring. Each transaction is labeled and the score is plotted in temporal order.

![Fraud Score History Visualisation](assets/show_fraud_history.png)

---

### 🧾 Keeping Records on Transaction History

After each transaction is submitted, it is added to an interactive table showing:
- Summary description
- Assigned fraud score
- Risk level

This feature is useful for auditors, model debugging, or dashboard integration.

![Transaction History](assets/show_transaction_history.png)

---

### 📖 Detailed Breakdown on the Fraud Score

Each individual prediction is backed by reason codes extracted from rule-based logic and LLM-enhanced logic (if enabled). These reasons ensure interpretability of the model’s decision.

Example reasons may include:
- High amount
- New device
- Foreign location mismatch
- Unusual transaction hour
- High-risk country

![Detailed Breakdown](assets/detailed_explanation.png)

---

## 🤗 4. Transformer-Based LLM Tools (`transformers_llm/`)

Reusable wrapper around HuggingFace Transformers for:

- Tokenization, logits, and pipeline usage
- Can be extended to support RAG-style reasoning and GPT-based response generation

---

## ✅ Running the Project

```bash
# Clone and install
git clone https://github.com/stevokwon/genai_fasttrack.git
cd genai_fasttrack
pip install -r requirements.txt
```

### 🧪 Run DCGAN

```bash
python genai_rl_agents/dcgan_mnist.py
```

### 🧪 Run Fraud Agent

```bash
python genai_rl_agents/fraud_risk_agent.py
```

### 🧪 Launch Streamlit Web App

```bash
streamlit run genai_rl_agents/fraud_risk_streamlit.py
```

### 🧪 Run Hugging Face Inference

```bash
python transformers_llm/huggingface_inference.py
```
