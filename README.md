# LLM-Based Text Watermark Detection using Llama 3.2 + NLP

An experimental NLP pipeline for detecting text generated with a deterministic lexical watermark signal using **Ollama + Llama 3.2**, watermark statistics, stylometric features, TF-IDF/n-grams, and a machine-learning classifier.

> **Important:** The supplied JSONL dataset contains `id` and `text`, but no watermark labels. Therefore, the notebook creates an experimental synthetic detection dataset from Llama 3.2 outputs. The reported metrics describe this experimental setup and should not be interpreted as production-grade watermark detection performance.

## Project Architecture

![Project Architecture](docs/watermarked_pipeline.jpg)

**Pipeline:**

`JSONL → Preprocessing → Ollama/Llama 3.2 → Normal + Watermark-oriented generation → NLP Feature Extraction → Feature Fusion → Classifier → Evaluation → Final Detector`

### Main components

1. **Data preprocessing**
   - Loads JSONL records.
   - Removes URLs/HTML and normalizes whitespace.
   - Computes basic word/character statistics.

2. **Llama 3.2 generation**
   - Uses Ollama with the `llama3.2` model.
   - Generates normal and watermark-oriented text.

3. **Deterministic watermark signal**
   - Uses a secret key and SHA-256 hashing to assign words to a deterministic green list.
   - Calculates green-word ratio and watermark z-score.

4. **NLP feature extraction**
   - Watermark statistics: z-score, green ratio, word count.
   - Stylometry: word count, average word length, vocabulary uniqueness, sentence length, punctuation, capitalization.
   - TF-IDF with unigram and bigram features, up to 5,000 features.

5. **Feature fusion**
   - Combines sparse TF-IDF features with numerical watermark/stylometric features.

6. **Classifier**
   - Logistic Regression with balanced class weights.

7. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 score
   - ROC-AUC
   - Classification report
   - Confusion matrix

8. **Inference**
   - A `final_detector()` function returns:
     - prediction
     - probability
     - z-score
     - green ratio
     - total words

## Repository Structure

```text
llama32-nlp-watermark-detection/
├── Llama32_NLP_Text_Watermark_Detection.ipynb
├── data/
│   └── train.jsonl
├── docs/
│   └── watermarked_pipeline.jpg
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.9+
- Ollama
- Llama 3.2 model

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then pull the model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before executing the generation cells in the notebook.

## Run the Project

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Llama32_NLP_Text_Watermark_Detection.ipynb
```

Run the cells from top to bottom.

## GitHub

After creating a new GitHub repository, run:

```bash
cd llama32-nlp-watermark-detection
git init
git add .
git commit -m "Add Llama 3.2 NLP watermark detection project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your new GitHub repository.

## Limitations

This is an experimental detector, not a production watermark detector. The source dataset has no ground-truth watermark labels, so labels are generated from the experimental normal/watermark-oriented generation process.

For a research-grade evaluation, use independently verified ground-truth labels and evaluate across human-written text, unwatermarked LLM text, genuinely watermarked LLM text, and paraphrased/adversarial text.

## Technologies

Python · NLP · Llama 3.2 · Ollama · TF-IDF · N-grams · Stylometry · Scikit-learn · Logistic Regression · Jupyter Notebook
