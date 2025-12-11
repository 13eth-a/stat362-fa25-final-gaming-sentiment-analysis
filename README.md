# Gaming Sentiment Analysis

A comprehensive machine learning project that analyzes sentiment in video game reviews using multiple model architectures, from simple baselines to state-of-the-art transformers.

## Project Overview

This project implements and compares various machine learning approaches for sentiment classification on Amazon video game reviews, progressing from simple baselines to sophisticated deep learning models:

- **Simple Baseline**: Logistic Regression with TF-IDF features
- **MLP**: Multi-Layer Perceptron with sentence embeddings for testing basic NN capacity 
- **RNN Models**: GRU and BiGRU with word-level embeddings for capturing temporal dependencies and sequential patterns
- **Transformer**: Fine-tuned DistilBERT for stronger contextual understanding

## Dataset

- **Source**: [Amazon Video Game Reviews](https://amazon-reviews-2023.github.io) (150K samples)
- **Classes**: 3-class sentiment (Negative, Neutral, Positive)
- **Features**: Review text, ratings, user information, timestamps
- **Split**: 80% Train / 10% Validation / 10% Test
- **Preprocessing**: Text cleaning, sentiment mapping (1-2 stars→Negative, 3 stars→Neutral, 4-5 stars→Positive)

## Repository Structure

```
├── data/
│   ├── raw/                    # Original dataset files
│   ├── processed/              # Cleaned and split data
│   └── embeddings/             # Pre-computed embeddings
├── notebooks/
│   ├── 01_data_prep.ipynb      # Data exploration and preprocessing
│   ├── 02_simple_baseline.ipynb # Logistic Regression baseline
│   ├── 03_mlp.ipynb            # Multi-Layer Perceptron
│   ├── 04_rnn_models.ipynb     # GRU and LSTM models
│   ├── 05_distilbert.ipynb     # Transformer fine-tuning
│   └── results/                # Generated figures and metrics
│       ├── figures/            # Visualizations and plots
│       └── metrics/            # JSON files with model results
├── models/                     # Saved trained models
├── results/                    # Final results and visualizations
└── requirements.txt            # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab account (recommended for GPU access)
- ~5GB storage for datasets and models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/13eth-a/gaming-sentiment-analysis.git
cd gaming-sentiment-analysis
```

2. Install the project and dependencies:

**Option A: Modern approach (recommended)**
```bash
pip install -e .                    # Core dependencies only
pip install -e ".[dev]"             # Include Jupyter for development
pip install -e ".[all]"             # All optional dependencies
```

**Option B: Traditional approach**
```bash
pip install -r requirements.txt     # Install exact versions from requirements
```

3. Download the dataset:
   - Place the [Amazon Video Games dataset](https://amazon-reviews-2023.github.io) (`Video_Games.jsonl.gz`) in `data/raw/`

### Usage

Run the notebooks in order:

1. **Data Preparation**: `01_data_prep.ipynb` - Loads, cleans, and preprocesses the dataset
2. **Baseline Model**: `02_simple_baseline.ipynb` - Establishes performance baseline with Logistic Regression
3. **Neural Networks**: `03_mlp.ipynb` - Multi-Layer Perceptron with embeddings
4. **Sequence Models**: `04_rnn_models.ipynb` - GRU and BiGRU architectures for text analysis
5. **Transformers**: `05_distilbert.ipynb` - Fine-tuned DistilBERT model

## Model Architectures

### 1. Simple Baseline (Logistic Regression)
- **Features**: TF-IDF vectors (10K features)
- **Purpose**: Establishes minimum performance threshold
- **Training Time**: ~30 seconds

### 2. Multi-Layer Perceptron (MLP)
- **Input**: 384-dimensional sentence embeddings (SentenceTransformers)
- **Architecture**: Dense layers with dropout and batch normalization
- **Training Time**: ~5 minutes

### 3. RNN Models (GRU/LSTM)
- **Input**: Word-level embeddings (50 tokens × 384 dimensions)
- **Architecture**: Bidirectional RNNs with attention mechanisms
- **Training Time**: ~20 minutes

### 4. DistilBERT (Transformer)
- **Input**: Tokenized text (max 128 tokens)
- **Architecture**: Pre-trained transformer fine-tuned on gaming reviews
- **Training Time**: ~45 minutes

## Results

| Model | Accuracy | Macro F1 | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
| Baseline (LogReg) | 77.3% | 0.624 | ~30K | <1m |
| MLP | 73% | 0.572 | ~850K | ~5m |
| GRU | 73% | 0.593 | ~1.2M | ~20m |
| BiGRU | 76% | 0.608 | ~1.5M | ~25m |
| DistilBERT | 84% | 0.667 | 67M | ~45m |

### Key Findings
- Positive reviews dominate (~78%), requiring balanced class weights for fair evaluation
- Logistic Regression with TF-IDF achieves 77.3% accuracy as a strong baseline
- MLP shows lower performance (73.3%) possibly due to overfitting or insufficient training
- DistilBERT achieves highest performance (84%) due to stronger contextual understanding enabled by self-attention; also has the longest training time
- All models struggle most with neutral sentiment classification (class 1)
- Results demonstrate the evolution from traditional ML to deep learning approaches


## Available Results and Visualizations

The repository includes comprehensive results for completed models:

### Metrics Files (`notebooks/results/metrics/`)
- `baseline_results.json` - Complete evaluation metrics for Logistic Regression baseline
- `mlp_results.json` - Detailed performance data for MLP model
- Results for GRU, BiGRU, and DistilBERT models available in notebook outputs

### Visualizations (`notebooks/results/figures/`)
- Confusion matrices showing classification patterns
- Per-class performance comparisons
- Training curves for neural network models
- Data distribution plots from preprocessing phase

### Performance Breakdown
**Baseline (Logistic Regression)**:
- Overall Accuracy: 77.3%
- Per-class F1: Negative (67.1%), Neutral (32.2%), Positive (88.1%)
- Strong performance on positive class, struggles with neutral reviews

**MLP Model**:
- Overall Accuracy: 73.3%
- Per-class F1: Negative (60.7%), Neutral (26.2%), Positive (84.8%)
- Shows similar pattern but with lower overall performance

**GRU Model**:
- Overall Accuracy: 73.1%
- Macro F1: 0.593
- Similar performance to MLP but with sequential processing capabilities
- Slightly better balanced performance across classes

**Bidirectional GRU (BiGRU)**:
- Overall Accuracy: 76.2%
- Macro F1: 0.608
- Best performing RNN model due to bidirectional context
- Approaches baseline performance with deeper understanding of sequence structure

**DistilBERT (Transformer)**:
- Overall Accuracy: 83.7%
- Macro F1: 0.667
- Best overall performance with significant improvement over other models
- Superior contextual understanding through transformer attention mechanisms
- 7+ percentage point improvement over baseline despite smaller training sample

## Technical Details

### Hardware Requirements
- **Training**: Google Colab with T4 GPU (16GB VRAM)
- **Inference**: CPU-only deployment possible for smaller models
- **Storage**: ~3GB for datasets, embeddings, and trained models

### Key Technologies
- **Deep Learning**: TensorFlow 2.19, PyTorch
- **NLP**: HuggingFace Transformers, SentenceTransformers
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn

### Data Processing Pipeline
1. **Text Cleaning**: Remove special characters, normalize whitespace
2. **Sentiment Mapping**: Convert 5-star ratings to 3-class labels
3. **Stratified Sampling**: Maintain class distribution across splits
4. **Class Balancing**: Compute balanced weights for training
5. **Embedding Generation**: Pre-compute embeddings for efficiency

## Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **Macro F1-Score**: Balanced performance across all classes
- **Per-Class Metrics**: Precision, Recall, F1 for each sentiment
- **Confusion Matrices**: Detailed misclassification analysis
- **Training Curves**: Loss and accuracy progression



## License

This project is for educational purposes as part of STAT 362 coursework.

## Acknowledgments

- [Amazon Customer Reviews Dataset](https://amazon-reviews-2023.github.io)
- HuggingFace Transformers library
- Google Colab for computational resources
- STAT 362 course materials and guidance

---

