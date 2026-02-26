# 📄 Hybrid Text Summarizer Using LLM

**BART-large-CNN + Groq / Gemini Hybrid System**
**Akashchand Rajput — AI & Data Science Student, AIDTM College**
Gandhinagar, Gujarat, India · Feb 2026

---

## 📌 Project Overview

This project implements a hybrid text summarization system that combines traditional transformer-based summarization with modern Large Language Models (LLMs).

It demonstrates how real-world NLP systems integrate:

* Extractive summarization
* Abstractive refinement
* Web scraping pipelines
* Multi-document processing
* Topic classification
* Evaluation using ROUGE-2

The system is built as an academic project showcasing practical AI system design.

---

## 🎯 Why This Project Was Built

Modern news consumption involves large volumes of information from multiple sources. Manually reading and synthesizing articles is time-consuming.

This system aims to:

* Automatically extract content from news websites
* Produce concise fact-preserving summaries
* Combine multiple documents into a unified briefing
* Demonstrate hybrid NLP techniques used in industry

---

## 🧠 Hybrid Summarization Pipeline

```text
Input Text / News Article
        ↓
BART-large-CNN (compression)
        ↓
LLM Refinement (Groq or Gemini)
        ↓
Final Fact-Preserving Summary
```

### ✅ Why a Hybrid Approach?

Pure extractive models often lack readability, while pure LLM summaries may hallucinate facts.

Combining both provides:

* Structure + factual grounding (BART)
* Fluency + coherence (LLM)

---

## 🔍 Why Each Technology Was Used

### 🧾 BART-large-CNN (Base Summarizer)

**Why used:**

* State-of-the-art transformer for summarization
* Pretrained specifically on news articles
* Produces reliable, structured summaries
* Works locally (no API required)
* Reduces input size before LLM processing

**Why not alternatives:**

* T5 → Slower and requires careful prompting
* PEGASUS → Heavy and less accessible on CPU
* GPT-style models → Require API, not offline capable
* Extractive algorithms (TextRank, LexRank) → Lower quality

---

### 🤖 Groq Llama-3.1-8B (Fast LLM Option)

**Why used:**

* Extremely fast inference
* Good factual consistency
* Free tier available
* Suitable for real-time applications

**Why not larger models:**

* Larger LLMs increase latency
* Higher cost
* Not necessary for refinement tasks

---

### ✨ Google Gemini 1.5 Flash (Reasoning-Focused Option)

**Why used:**

* Strong reasoning capabilities
* Good language fluency
* Handles long context efficiently
* Alternative provider for comparison

**Why not Gemini Pro / larger versions:**

* Higher latency
* Greater resource usage
* Flash model is optimized for speed-quality balance

---

### 🌐 Web Scraping (Requests + BeautifulSoup)

**Why used:**

* Works across most websites
* Lightweight and flexible
* No browser automation required
* Suitable for structured content extraction

**Why not Selenium or browser automation:**

* Much slower
* Requires heavy dependencies
* Unnecessary for static news pages
* Not ideal for lightweight academic deployment

---

### 🧩 Domain-Specific Selectors + Fallbacks

**Why used:**

News sites use different HTML structures.

Solution implemented:

1. Site-specific selectors (high accuracy)
2. Generic selectors (broad coverage)
3. JSON-LD structured data extraction
4. Microdata fallback

This layered approach improves extraction success.

---

### 📚 Multi-Document Processing

**Why implemented:**

Real-world scenarios often involve multiple sources.

Capabilities:

* Individual summaries
* Combined executive summary
* Cross-document insights

This simulates real analyst workflows.

---

### 🏷️ News Classification via LLM

**Why used:**

Traditional classifiers require labeled datasets.

LLMs can perform zero-shot classification using natural language prompts, making them ideal for flexible categorization without additional training.

---

## 🌐 Advanced Web Scraping Features

Supports major Indian and international news publishers.

Key capabilities:

* 120+ CSS selector patterns
* JSON-LD structured data extraction
* Noise filtering (ads, navigation, spam)
* Multi-article homepage processing
* Exponential backoff for failed requests

---

## 📊 Evaluation Metrics (Static)

Metrics displayed in the application are **pre-computed and static**, not calculated in real time.

### ✅ ROUGE-2 (Bigram Overlap)

Chosen because:

* Standard metric for summarization evaluation
* Measures content similarity at phrase level
* Widely used in research papers

Example benchmark values:

| Model Variant | ROUGE-2 |
| ------------- | ------- |
| BART Only     | 0.212   |
| BART + Groq   | 0.235   |
| BART + Gemini | 0.248   |

Additional reported estimates:

* Fact preservation
* Numeric accuracy
* Inference time

---

## 🖥️ Application Modules

### 1️⃣ Web Scraping Module

* Input news homepage URL
* Extract multiple articles
* Generate summaries and categories

---

### 2️⃣ Multi-Document Module

* Add custom documents
* Produce individual summaries
* Generate combined executive summary

---

### 3️⃣ Dataset Demo (XSum)

* Uses BBC XSum dataset samples
* Compare generated summaries with reference summaries

---

### 4️⃣ Evaluation Panel

* Displays static performance metrics
* Illustrates benefits of hybrid approach

---

## 🏗️ System Architecture

```text
News URL / Input Text
        ↓
Link Discovery & Scraping
        ↓
Content Extraction & Cleaning
        ↓
BART-large-CNN Summarizer
        ↓
LLM Refinement (Groq / Gemini)
        ↓
Classification & Output
        ↓
Streamlit Dashboard
```

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.10+
* Minimum 8 GB RAM
* Internet connection for model download

---

### Create Virtual Environment

```bash
python -m venv textsum
textsum\Scripts\activate.bat
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

### Optional API Keys

For LLM refinement:

* Groq → https://console.groq.com
* Gemini → https://aistudio.google.com

Without keys, the system falls back to BART-only summarization.

---

## ⚙️ Configuration Options

Sidebar controls allow:

* LLM provider selection
* Maximum articles to scrape
* Summary length (Short / Balanced / Detailed)
* API key input

---

## 🧪 Use Cases

* News aggregation and briefing
* Research summarization
* Academic NLP demonstrations
* Information overload reduction
* Multi-source analysis

---

## 📦 Requirements

Core libraries:

```
torch
transformers
streamlit
rouge-score
groq
google-generativeai
beautifulsoup4
requests
datasets
```

---

## 🔧 Limitations

* Metrics are static (not live evaluation)
* Scraping success depends on website structure
* Requires internet for LLM features
* Very long articles are truncated

---

## 👨‍🎓 Author

**Akashchand Rajput**
AI & Data Science Student
AIDTM College
Gandhinagar, Gujarat, India

---

## 📄 License

MIT License — Free for academic and personal use.

---

## ⭐ Acknowledgements

* Hugging Face Transformers
* Meta BART model
* Groq LLM API
* Google Gemini API
* BBC XSum Dataset
* Streamlit Framework

---

*This project demonstrates how modern NLP systems combine traditional transformer models with LLMs to achieve high-quality summarization.*
