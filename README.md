# Sentiment Analysis on Amazon Product Reviews

Machine Learning & Deep Learning project for sentiment classification on Amazon product reviews.

## Project Overview

This project implements and compares traditional machine learning models with deep learning models for sentiment analysis on Amazon product reviews. The goal is to classify review sentiments based on the review text and analyze which approach performs better.

**Dataset**: Amazon Product Reviews (568,454 reviews)
**Source**: [Kaggle - Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
**Date Range**: 1999-10-08 to 2012-10-26

**Best Result**: **90.58% Test Accuracy** (LSTM + GloVe, Binary Classification)

## Objectives

-   Implement at least one traditional ML model (Logistic Regression, SVM, or Naïve Bayes)
-   Implement at least one deep learning model (RNN, LSTM, or GRU)
-   Conduct comprehensive EDA with 4+ visualizations
-   Perform hyperparameter tuning with documented experiments
-   Include at least TWO experiment tables analyzing different hyperparameters
-   Evaluate models using appropriate metrics

## Project Structure

```
ml_sentiment_analysis/
 Notebook_Sentiment.ipynb # Main notebook (EDA, Models, Experiments)
 requirements.txt # Project dependencies
 README.md # This file
```

## Setup Locally

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)
-   4GB+ RAM recommended
-   2GB free disk space (for dataset and models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Clint07-datascientist/ML_Techniques_1_Sentiment_Analysis
cd ml_sentiment_analysis
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**

-   tensorflow (for LSTM model)
-   scikit-learn (for Logistic Regression)
-   pandas, numpy (data processing)
-   spacy (lemmatization)
-   kagglehub (dataset download)
-   contractions (text preprocessing)
-   matplotlib, seaborn, wordcloud (visualization)

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Download GloVe Embeddings

The notebook will automatically download GloVe embeddings (~862MB) on first run. Alternatively, download manually:

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### Step 6: Run the Notebook

**Option A: Jupyter Notebook**

```bash
jupyter notebook Notebook_Sentiment_A.ipynb
```

**Option B: JupyterLab**

```bash
jupyter lab
```

**Option C: Google Colab**

-   Upload `Notebook_Sentiment_A.ipynb` to [Google Colab](https://colab.research.google.com/)
-   Run all cells (Runtime → Run all)
-   All dependencies will be installed automatically

### Step 7: Expected Runtime

-   Data loading: ~2 minutes
-   Preprocessing: ~5-10 minutes (with caching)
-   Logistic Regression training: ~1-2 minutes
-   LSTM training: ~2-3 hours (without GPU)
-   **With GPU (Colab/Local)**: ~30-45 minutes

### Troubleshooting

**Issue: spaCy model not found**

```bash
python -m spacy download en_core_web_sm
```

**Issue: TensorFlow installation error**

```bash
pip install tensorflow --upgrade
```

**Issue: Out of memory during training**

-   Reduce `SAMPLE_SIZE_PER_CLASS` from 20000 to 10000
-   Close other applications
-   Use Google Colab for free GPU access

**Issue: Dataset download fails**

-   Check internet connection
-   Kaggle authentication may be required
-   Dataset will auto-download via kagglehub

---

## Quick Start (For Reviewers)

```bash
jupyter notebook Notebook_Sentiment_A.ipynb
```

**Note**: The notebook will automatically download the dataset from Kaggle on first run.

## Methodology

### 1. Exploratory Data Analysis

-   Distribution analysis of ratings
-   Review length analysis
-   Temporal trends
-   Correlation analysis
-   Word cloud visualization
-   10+ comprehensive visualizations

### 2. Preprocessing

-   Missing value handling
-   Text lowercasing
-   HTML tag removal & URL removal
-   Special character removal
-   Contraction expansion
-   spaCy lemmatization with stopword removal
-   Caching mechanism for faster reruns

### 3. Feature Engineering

-   **Binary Classification**: Dropped 3-star (neutral) reviews for clearer sentiment
-   **Text + Summary Concatenation**: Combined both fields for richer features
-   **Balanced Sampling**: 20K reviews per class (40K total)
-   TF-IDF vectorization (50,000 features, bigrams)
-   LSTM tokenization (sequence length: 200)
-   Train-test split (70-30, stratified)

### 4. Model Implementation

**Traditional ML Models**:

-   **Logistic Regression** (C=1.0) - **87-89% accuracy**

**Deep Learning Models**:

-   **LSTM + GloVe** (Bi-LSTM 128 units, frozen embeddings) - **90.58% accuracy**

### 5. Evaluation Metrics

-   Accuracy
-   Precision, Recall, F1-Score
-   Confusion Matrix
-   Classification Report
-   Training Time

## Experiment Results

### Experiment 1: Logistic Regression (3-Class Sentiment)

| Regularization (C) | Accuracy   | Training Time (s) | Notes                                   |
| ------------------ | ---------- | ----------------- | --------------------------------------- |
| **1.0** (default)  | **73.71%** | 4.46s             | Best balance of accuracy and speed      |
| 10.0 (weak reg)    | 73.49%     | 7.06s             | Slightly slower, marginal accuracy loss |
| 50.0 (very weak)   | 72.09%     | 8.15s             | Overfitting signs, slower convergence   |
| 0.1 (strong reg)   | 69.62%     | 3.11s             | Fastest but underfits                   |

**Fixed Parameters**: solver=lbfgs, max_iter=2000, class_weight=balanced, 50K TF-IDF features (1-2 grams)

**Key Findings**:

-   Default regularization (C=1.0) achieves optimal performance
-   Strong regularization (C=0.1) causes underfitting (-4.1% accuracy)
-   Weak regularization increases training time without benefit
-   TF-IDF with bigrams effective for sentiment classification

---

### Experiment 2: LSTM Baseline (3-Class Sentiment - NLTK, 15K/class)

| Configuration | Batch Size | LSTM Units    | Embedding Dim | Test Accuracy | Val Accuracy | Training Time |
| ------------- | ---------- | ------------- | ------------- | ------------- | ------------ | ------------- |
| Config 1      | 32         | 128 (Bi-LSTM) | 200 (learned) | 68.98%        | 67.43%       | 52.0 min      |
| Config 2      | 64         | 128 (Bi-LSTM) | 200 (learned) | **69.08%**    | 67.54%       | 38.4 min      |
| Config 3      | 128        | 128 (Bi-LSTM) | 200 (learned) | 68.60%        | 67.73%       | 28.1 min      |

**Fixed Parameters**: NLTK lemmatization, Adam (lr=0.001), Dropout=0.3/0.5, SpatialDropout1D=0.2

**Findings**: LSTM underperforming Logistic Regression (69% vs 74%) - weak embeddings

---

### Experiment 3: LSTM Optimized (spaCy + GloVe, 20K/class) - OVERFITTING DETECTED

| Configuration | Preprocessing | Embeddings       | Trainable | Test Acc | Val Acc | Train Acc (Final) | Training Time | Epochs |
| ------------- | ------------- | ---------------- | --------- | -------- | ------- | ----------------- | ------------- | ------ |
| **Config A**  | spaCy         | Learned 300d     | Yes       | 68.37%   | 68.45%  | ~93%              | 113.5 min     | 7      |
| **Config B**  | spaCy         | GloVe Frozen     | No        | ~68.5%   | 68.51%  | ~74%              | ~120 min      | 11     |
| **Config C**  | spaCy         | GloVe Fine-tuned | Yes       | ~69.9%   | 69.88%  | ~92%              | ~120 min      | 8      |

**Fixed Parameters**: Bi-LSTM 128 units, Dropout=0.3/0.5, SpatialDropout1D=0.2, EarlyStopping (patience=5), 60K samples

**CRITICAL FINDINGS - Severe Overfitting**:

-   **Config A**: Training acc ~93% vs Test acc 68.37% → **~25% overfitting gap**
-   **Config B**: Training acc ~74% vs Test acc 68.5% → **~6% gap** (better, frozen helps)
-   **Config C**: Training acc ~92% vs Test acc 69.9% → **~22% overfitting gap**
-   **NO improvement over baseline** despite spaCy + GloVe + more data
-   **All configs memorizing training data** instead of learning patterns
-   **Config B (frozen) shows least overfitting** - embeddings act as regularization

---

## Overfitting Analysis & Solutions

### Root Causes Identified:

1. **Too Much Model Capacity** - 17.5M parameters for 60K samples (1:3.4 ratio)
2. **Insufficient Regularization** - Current dropout (0.3/0.5) not strong enough
3. **Early Stopping Too Late** - Patience=5 allows too much overfitting
4. **No L2 Regularization on LSTM** - Only on Dense layer
5. **Learning Rate Too High** - 0.001 allows rapid memorization

### Anti-Overfitting Strategies to Implement:

| Technique                      | Current | Recommended | Expected Impact       |
| ------------------------------ | ------- | ----------- | --------------------- |
| **Dropout Rate**               | 0.3/0.5 | 0.5/0.6     | Reduce overfitting 5% |
| **L2 Regularization (LSTM)**   | None    | 1e-3        | Reduce overfitting 5% |
| **Early Stopping Patience**    | 5       | 2-3         | Stop before overfit   |
| **Learning Rate**              | 0.001   | 0.0005      | Slower, stable learn  |
| **Batch Normalization**        | No      | Yes         | Regularization +3%    |
| **Reduce LSTM Units**          | 128     | 64          | Less capacity         |
| **Data Augmentation**          | No      | Synonyms    | +2-3% generalization  |
| **Ensemble (Dropout at test)** | No      | Yes         | +1-2% accuracy        |

### Experiment 4: Anti-Overfitting LSTM (RECOMMENDED)

**Configuration D: Heavily Regularized**

-   **Increased Dropout**: 0.5 (embeddings), 0.6 (LSTM), 0.6 (Dense)
-   **L2 Regularization**: All layers (1e-3)
-   **Reduced Capacity**: 64 LSTM units (vs 128)
-   **Lower Learning Rate**: 0.0005 (vs 0.001)
-   **Early Stopping**: Patience=2 (vs 5)
-   **Batch Normalization**: After LSTM layer
-   **GloVe Frozen**: Prevent embedding overfitting

**Expected Results:**

-   Target Gap: <10% (train-test accuracy difference)
-   Target Test Accuracy: 72-75% (better generalization)
-   Training Time: ~40-50 min (stops earlier)

### Experiment 5: Minimal Model + Batch Normalization

**Configuration E: Alternative Anti-Overfitting**

-   **Minimal Capacity**: 32 LSTM units (vs 128)
-   **Batch Normalization**: After LSTM (regularization effect)
-   **Moderate Dropout**: 0.4
-   **Standard Learning Rate**: 0.001
-   **Light L2**: 1e-4
-   **GloVe Frozen**: Prevent embedding overfitting

**Philosophy**: "Less is more" - smallest model that can learn

**Expected Results:**

-   Target Gap: <8% (better than Config D)
-   Target Test Accuracy: 70-73%
-   Training Time: ~30-40 min (fastest, fewer parameters)

---

## Summary of All Configurations

### Text-Only Experiments (A-H): All Failed to Reach 75%+

| Config | Strategy           | Features  | LSTM | Dropout | Train Acc | Val Acc | Test Acc | Gap   | Time    | Status   |
| ------ | ------------------ | --------- | ---- | ------- | --------- | ------- | -------- | ----- | ------- | -------- |
| A      | Learned embeddings | Text only | 128  | 0.3/0.5 | ~93%      | 68.4%   | 68.4%    | ~25%  | 113 min | Overfit  |
| B      | GloVe frozen       | Text only | 128  | 0.3/0.5 | ~74%      | 68.5%   | 68.5%    | ~6%   | 120 min | Low      |
| C      | GloVe fine-tuned   | Text only | 128  | 0.3/0.5 | ~92%      | 69.9%   | 69.9%    | ~22%  | 120 min | Overfit  |
| D      | Heavy reg          | Text only | 64   | 0.6     | 56.9%     | 62.3%   | 61.8%    | -4.9% | 67 min  | Underfit |
| E      | Minimal + BN       | Text only | 32   | 0.4     | 65.3%     | 66.7%   | 66.9%    | -1.6% | 67 min  | Too weak |
| F      | Balanced           | Text only | 48   | 0.5     | 63.0%     | 65.8%   | 65.5%    | -2.5% | 85 min  | Low      |
| H      | GloVe optimized    | Text only | 64   | 0.4     | 87.3%     | 69.1%   | 69.1%    | 18.2% | 67 min  | Overfit  |

### CRITICAL PROBLEM DISCOVERED: Missing Features!

**Root Cause Analysis:**

1. **Only using `Text` column** → Ignoring `Summary` (review titles with concentrated sentiment!)
2. **Ignoring `Helpfulness` data** → Missing quality/sentiment correlation signal
3. **Result**: Stuck at 66-69% accuracy, way too low for 3-class with 60K samples

**Dataset Columns Available:**

-   `Text` - Full review text (USED)
-   `Summary` - Review title/summary (NOT USED - BIG MISTAKE!)
-   `HelpfulnessNumerator` / `HelpfulnessDenominator` (NOT USED!)
-   `Score`, `Time`, `ProductId`, etc.

### NEW APPROACH: Feature Engineering + Quality Filtering

| Config      | Features             | Filter Strategy               | Architecture | LSTM | GloVe   | Test Acc   | Val Acc | Train Acc | Gap   | Time    | Status         |
| ----------- | -------------------- | ----------------------------- | ------------ | ---- | ------- | ---------- | ------- | --------- | ----- | ------- | -------------- |
| **SIMPLE**  | Text+Summary         | None (all reviews)            | Sequential   | 96   | Frozen  | TBD        | TBD     | TBD       | TBD   | ~60 min | Ready to run   |
| **SMART-A** | Text+Summary         | Voted only (Denom > 0)        | Sequential   | 96   | Frozen  | **72.86%** | 73.93%  | 73.15%    | 0.29% | 152 min | Best so far!   |
| **SMART-B** | Text+Summary         | Quality (>=3v, >=60% helpful) | Sequential   | 96   | Frozen  | 72.28%     | 70.71%  | 74.51%    | 2.23% | 104 min | Less data hurt |
| **MULTI-A** | Text+Sum+Helpfulness | None                          | Multi-Input  | 80   | Frozen  | ~70%       | ~73%    | ~72%      | ~2%   | ~180min | Training       |
| **MULTI-B** | Text+Sum+Helpfulness | None                          | Multi-Input  | 96   | 2-Phase | TBD        | TBD     | TBD       | TBD   | TBD     | Training       |

**Key Findings:**

1. **Text + Summary Concatenation**: **+3% accuracy** (69.9% → 72.86%)
2. **Quality Filtering (SMART-A)**: Best results with voted reviews (52% of data)
3. **Too Aggressive Filtering (SMART-B)**: Loses too much data (~17% kept), doesn't improve
4. **Multi-Input Complexity**: Not significantly better than simple concatenation
5. **Winner: Config SMART-A** → 72.86% test, 73.93% val, minimal overfitting (0.29%)

---

## BREAKTHROUGH INSIGHT: The Neutral Class Problem

### Root Cause Analysis

**After 15+ experiments, we discovered the fundamental issue:**

-   **3-Star reviews (Neutral) are inherently ambiguous**
-   **Stuck at 72-73% accuracy ceiling across ALL configurations**
-   **Problem**: 3-star reviews contain mixed sentiment (weak positives, disappointed expectations, "it's okay")

**Dataset Analysis:**

| Score | Original Label | % of Dataset | Sentiment Clarity |
| ----- | -------------- | ------------ | ----------------- |
| 1-2   | Negative       | 14.4%        | Clear             |
| 3     | **Neutral**    | 7.5%         | **Ambiguous**     |
| 4-5   | Positive       | 78.1%        | Clear             |

**Examples of Ambiguous 3-Star Reviews:**

-   "It's okay, nothing special" (True Neutral)
-   "Good but overpriced" (Positive sentiment, negative value)
-   "Expected more" (Disappointed = Negative-leaning)
-   "Works fine" (Minimal Positive)

### SOLUTION: Binary Classification (Negative vs Positive)

**Recommended Next Experiment:**

| Experiment            | Classification | Data                  | Expected Accuracy | Status      |
| --------------------- | -------------- | --------------------- | ----------------- | ----------- |
| Experiment 6 (BINARY) | **2-Class**    | Drop Score=3 reviews  | **85-90%**        | Recommended |
| Current Best          | 3-Class        | All reviews (SMART-A) | 72.86%            | Completed   |

**Implementation Plan:**

1. **Data Preparation:**

-   Filter dataset: `df = df[df['Score'] != 3]`
-   New labels: `Score 1-2 → 0 (Negative)`, `Score 4-5 → 1 (Positive)`
-   Use Config SMART-A strategy (voted reviews + Text+Summary)

2. **Model Changes (Simple!):**

```python
# Output layer change
Dense(1, activation='sigmoid') # Was: Dense(3, activation='softmax')

# Loss function change
loss='binary_crossentropy' # Was: loss='categorical_crossentropy'
```

3. **Expected Results:**

-   **Accuracy**: 85-90% (vs 72.86% current)
-   **Gain**: +12-17% by removing ambiguity
-   **Training Time**: Similar (~150 min)
-   **Rubric Compliance**: Still "Sentiment Analysis" (Task allows scope definition)

**Why This Works:**

-   Eliminates inherently ambiguous examples
-   Clear decision boundary (negative vs positive)
-   More actionable for business (3-star doesn't inform action)
-   Standard practice in industry (many companies use 2-class for sentiment)

**What to Do with Neutral Reviews Later:**

_Option 1_: Separate 3-class predictor for ambiguous cases
_Option 2_: Confidence thresholding (low confidence → "Neutral")
_Option 3_: Ignore them (7.5% of data, low business value)

---

### Model Comparison Summary

| Model                   | Best Config | Accuracy   | Training Time | Complexity |
| ----------------------- | ----------- | ---------- | ------------- | ---------- |
| **Logistic Regression** | C=1.0       | **73.71%** | 4.5s          | Low        |
| LSTM (Bi-LSTM 128)      | Batch=64    | 69.08%     | 38.4 min      | High       |

**Winner**: Logistic Regression (+4.6% accuracy, 500x faster)

---

## What Went Wrong & How We're Fixing It

### Critical Mistakes in Previous Experiments (A-H)

#### 1. **Feature Underutilization (BIGGEST MISTAKE)**

**What We Did Wrong:**

-   Only used the `Text` column from the dataset
-   **Completely ignored `Summary`** (review titles!) - these are sentiment-dense!
-   **Completely ignored `Helpfulness` data** - strong quality/sentiment signal

**Example from Dataset:**

```
Summary: "Great taffy"
Text: "Great taffy at a great price. There was a wide assortment..."
HelpfulnessNumerator: 0, HelpfulnessDenominator: 0
```

**Impact**: Missing 30-40% of available signal → Stuck at 66-69% accuracy

#### 2. **Severe Overfitting with Fine-tuned GloVe**

**What We Did Wrong (Config H):**

-   Fine-tuned GloVe with `trainable=True` from epoch 1
-   Used standard learning rate (0.001) for embedding layer
-   **Result**: Train 87.3%, Test 69.1% → **18.2% overfitting gap!**

**Why It Failed**: GloVe's pre-trained knowledge was destroyed by aggressive updates

#### 3. **Weak Word Embeddings in Early Experiments**

**What We Did Wrong (Configs A-C baseline):**

-   LSTM learns embeddings from only 45K reviews
-   Vocabulary of 52K words means many words have <10 occurrences
-   No pre-trained embeddings in Config A

**Impact**: LSTM lacks semantic understanding, leading to underfitting

---

### The Fix: Multi-Modal Input Architecture

#### **Config MULTI-A: Text+Summary+Helpfulness (GloVe Frozen)**

**What We're Doing Right:**

1. **Feature Engineering:**

-   `Combined_Text = Summary + " " + Text` → Concatenate both text fields
-   `Helpfulness_Ratio = Numerator / Denominator` → Numerical feature (0.0 if denom=0)

2. **Multi-Input Model Architecture:**

```python
Input 1: Text sequences → Embedding (GloVe Frozen) → Bi-LSTM → Vector A
Input 2: Helpfulness → Dense(16) → Vector B
Concatenate [Vector A, Vector B] → Dense → Softmax
```

3. **Why It Works:**

-   **Summary titles** are sentiment-concentrated (e.g., "Not as Advertised", "Great!", "Cough Medicine")
-   **Helpfulness ratio** correlates with strong sentiment (high-quality reviews)
-   **GloVe frozen** prevents 18% overfitting seen in Config H

**Expected Results:** 72-78% test accuracy (+6-9% improvement)

---

#### **Config MULTI-B: 2-Phase Gradual Fine-tuning**

**Advanced Strategy:**

**Phase 1 (7 epochs):** GloVe **FROZEN**, LR=0.001

-   Let LSTM weights adapt to high-quality GloVe features
-   Build stable representations

**Phase 2 (20 epochs):** GloVe **TRAINABLE**, LR=5e-5 (micro!)

-   Unfreeze GloVe layer carefully
-   Use microscopically small learning rate (100x smaller!)
-   Fine-tune without destroying pre-trained knowledge

**Why It Works:**

-   Prevents catastrophic forgetting of GloVe's semantic knowledge
-   Allows embeddings to adapt to Amazon review domain
-   Combines multi-modal features with smart fine-tuning

**Expected Results:** 75-80% test accuracy (+9-11% improvement)

---

#### 2. **Preprocessing Compromises (Minor Issue)**

Comparison between notebooks:

| Aspect           | Original (Notebook_A) | Current (Notebook_1) | Impact                   |
| ---------------- | --------------------- | -------------------- | ------------------------ |
| Lemmatization    | spaCy (contextual)    | NLTK (rule-based)    | Less accurate word forms |
| Stopword Removal | Applied               | Applied              | Maintained               |
| Vocabulary Size  | 52,419 words          | 52,419 words         | Same                     |
| Training Samples | 31,496                | 31,496               | Same                     |

**Finding**: Lemmatization quality may be limiting LSTM's ability to learn semantic patterns.

#### 3. **Limited Training Data for Embeddings**

-   45K samples insufficient to learn quality 200-dim embeddings
-   Vocabulary of 52K words means many words have <10 occurrences
-   Pre-trained embeddings (GloVe/Word2Vec) trained on billions of words

---

## Recommended Improvements (Prioritized)

### **HIGH PRIORITY - Embedding Strategy**

#### 1. **Implement Pre-trained GloVe Embeddings**

**Expected Gain**: +8-12% accuracy (target: 78-82%)

**Implementation**:

```python
# Use GloVe 300d (trained on 840B tokens)
- Download: glove.840B.300d.txt
- Initialize Embedding layer with GloVe weights
- Option A: Freeze embeddings (faster, good baseline)
- Option B: Fine-tune embeddings (slower, potentially better)
```

**Rationale**:

-   Provides rich semantic representations from massive corpus
-   Addresses vocabulary coverage issues
-   Standard practice for small-medium datasets

#### 2. **Try Word2Vec Embeddings**

**Expected Gain**: +6-10% accuracy

**Implementation**:

```python
# Google's Word2Vec (300d, trained on Google News)
- Use gensim to load pre-trained vectors
- Compare with GloVe performance
```

---

### **MEDIUM PRIORITY - Architecture & Preprocessing**

#### 3. **Restore spaCy Lemmatization**

**Expected Gain**: +1-2% accuracy

**Implementation**:

```python
# Revert to spaCy for better contextual lemmatization
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
```

**Rationale**: spaCy uses contextual rules, better handles:

-   Proper nouns (Amazon, Kindle)
-   Domain-specific terms (product names)
-   Part-of-speech disambiguation

#### 4. **Experiment with CNN-LSTM Hybrid**

**Expected Gain**: +2-4% accuracy

**Architecture**:

```python
Sequential([
 Embedding(vocab_size, 300, weights=[glove_matrix]),
 Conv1D(128, 5, activation='relu'), # Capture local patterns
 GlobalMaxPooling1D(),
 Dense(128, activation='relu'),
 Dropout(0.5),
 Dense(3, activation='softmax')
])
```

**Rationale**: CNNs excel at capturing n-gram patterns in text

#### 5. **Increase Sequence Length**

**Expected Gain**: +1-2% accuracy

**Current**: maxlen=100 tokens
**Recommended**: maxlen=150-200 tokens

**Rationale**: Amazon reviews average 436 chars (~70-90 tokens), but many longer reviews may be truncated

---

### **LOW PRIORITY - Fine-tuning**

#### 6. **Advanced LSTM Configurations**

-   Try stacked LSTM layers (2-3 layers)
-   Experiment with attention mechanism
-   Test different dropout rates (0.2, 0.4, 0.6)

#### 7. **Ensemble Methods**

-   Combine Logistic Regression + LSTM predictions
-   Weighted voting based on confidence scores

---

## Suggested Next Experiments

### Experiment 3: Embedding Strategies (CRITICAL)

| Config | Embedding Type    | Dimension | Trainable        | Expected Accuracy |
| ------ | ----------------- | --------- | ---------------- | ----------------- |
| A      | Learned (current) | 200       | Yes              | ~69% (baseline)   |
| B      | GloVe 300d        | 300       | No (frozen)      | **76-80%**        |
| C      | GloVe 300d        | 300       | Yes (fine-tuned) | **78-82%**        |
| D      | Word2Vec 300d     | 300       | No (frozen)      | 75-79%            |

### Experiment 4: Preprocessing Impact

| Config | Lemmatizer     | Stopwords | Max Length | Expected Accuracy |
| ------ | -------------- | --------- | ---------- | ----------------- |
| A      | NLTK (current) | Removed   | 100        | ~69% (baseline)   |
| B      | spaCy          | Removed   | 100        | 70-71%            |
| C      | spaCy          | Removed   | 150        | 71-72%            |
| D      | spaCy + GloVe  | Removed   | 150        | **77-81%**        |

---

## Current Progress

**Completed**:

-   Dataset loading and exploration (568K reviews)
-   3-class sentiment mapping (Negative/Neutral/Positive)
-   Balanced sampling (45K reviews, 15K per class)
-   10+ EDA visualizations
-   Text cleaning pipeline (NLTK-based)
-   Train-test split (70-30, stratified)
-   TF-IDF features (31.5K train × 50K features)
-   LSTM sequences (31.5K train × 100 maxlen)
-   **Experiment 1**: Logistic Regression tuning (4 configs)
-   **Experiment 2**: LSTM batch size testing (3/4 configs)

    **In Progress**:

-   LSTM Config 4 (Bi-LSTM 256 units)
-   Pre-trained embeddings implementation
-   Model performance analysis

    **Planned**:

-   **Experiment 3**: Embedding strategies (GloVe vs Word2Vec vs Learned)
-   **Experiment 4**: Preprocessing variations (spaCy vs NLTK)
-   Confusion matrices and error analysis
-   Final model selection and evaluation

## Key Insights from EDA

1. **Highly Imbalanced Dataset**: 63.9% of reviews are 5-stars
2. **Review Length**: Average 436 characters, median 302 characters
3. **Temporal Pattern**: Review volume increased significantly from 2010-2012
4. **Helpfulness**: Only 52.49% of reviews have helpfulness votes
5. **Correlation**: Higher ratings correlate with longer reviews

## Performance Optimizations

-   **Sampling Strategy**: Use 25K stratified sample for development (5K per rating)
-   **NLTK Lemmatization**: 10x faster than spaCy
-   **Caching System**: Processed data cached in `.parquet` files
-   **First Run**: ~2-3 minutes with lemmatization
-   **Subsequent Runs**: ~10 seconds (loads from cache)

## References

-   [Kaggle Dataset: Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
-   [scikit-learn Documentation](https://scikit-learn.org/)
-   [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
-   [NLTK Documentation](https://www.nltk.org/)

---

## Final Results

### Model Performance

| Model                   | Test Accuracy | Training Time | Complexity |
| ----------------------- | ------------- | ------------- | ---------- |
| **Logistic Regression** | 87-89%        | ~5-10s        | Low        |
| **LSTM + GloVe**        | **90.58%**    | ~120-150 min  | High       |

### Key Achievements

1. **Binary Classification Strategy**: Achieved **+17-18 percentage points** improvement over 3-class classification (72-73% → 90.58%)

2. **Feature Engineering Impact**:

-   Text + Summary concatenation: +3-5% accuracy
-   Balanced sampling: Removed class imbalance issues
-   spaCy lemmatization: Better word normalization

3. **Model Comparison**:

-   LSTM outperforms Logistic Regression by ~2-4%
-   Trade-off: LSTM is ~1000x slower to train
-   Both models benefit from binary classification

4. **Configuration from Config BINARY-SIMPLE**:

-   Bi-LSTM: 128 units
-   GloVe 300d embeddings (frozen)
-   Dropout: 0.35, L2 Reg: 1.5e-4
-   Minimal overfitting gap: <2%

### Business Value

-   **Product Quality Monitoring**: Identify declining sentiment trends
-   **Customer Feedback Analysis**: Prioritize negative reviews
-   **Actionable Insights**: Clear positive/negative classification
-   **Real-time Deployment**: Fast Logistic Regression for production

---

## License

This project is for academic purposes as part of coursework.

---

**Last Updated**: October 4, 2025
**Status**: COMPLETED - 90.58% Test Accuracy Achieved
**Team Size**: 4 members
**Best Model**: LSTM + GloVe (Binary Classification)
