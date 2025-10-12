# Hybrid Feature-Based Detection of AI-Generated Text

This repository implements a project exploring linguistic and statistical features for distinguishing human- vs. AI-generated text. We evaluate individual features (e.g., sentence complexity, punctuation, lexical diversity, perplexity, burstiness) and hybrid combinations/ensembles using lightweight classifiers like logistic regression and XGBoost. Hybrids often outperform single methods in robustness and generalization, as per recent surveys (Wu et al., 2025; Su & Wu, 2024).

## Research Question
Which linguistic or statistical features  are most effective in distinguishing human- vs. AI-generated text?

## Method
- Extract features using tools like language models (perplexity), tokenizers (lexical diversity), and POS taggers (ratios).
- Train classifiers on features; perform ablation studies and explore ensembles (e.g., stacking/voting).
- Analyze feature importances.

## Datasets
To generalize across sources/domains/LLMs and avoid overfitting:
- [*AI Text Detection Pile (Hugging Face)*](https://huggingface.co/datasets/artem9k/ai-text-detection-pile): ~1.3M samples, diverse texts (Reddit, Wikipedia, books) (Artem9k, 2023).
- [*AH&AITD (Arslan's Human and AI Text Database)*](https://figshare.com/articles/dataset/AH_AITD_Arslan_s_Human_and_AI_Text_Database/29144348?file=54804839): ~12k samples, texts across domains (articles, abstracts, stories, news, reviews) (Akram, 2023).
- [*LLM - Detect AI Generated Text Dataset (Kaggle sunilthite)*](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset): >28k essays (Thite, 2023).
- [*LLM - Detect AI Generated Text (Kaggle Competition)*](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv): ~10k essays (Kaggle, 2023).
- [*DAIGT V2 Train Dataset (Kaggle)*](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset?resource=download): 48k samples, essays (Deotte, 2023).
- [*Human ChatGPT Comparison Corpus (HC3)*](https://huggingface.co/datasets/Hello-SimpleAI/HC3): ~24k rows, Q&A from domains (finance, medicine, law, open) (Guo et al., 2023).

## Evaluation
Classifiers assessed via accuracy, F1-score, AUROC, and precision at false-positive rates. Feature importance via coefficients, SHAP values (Lundberg, 2017), or permutation importance. Hybrids benchmarked against baselines.

## Requirements
- Conda (for environment management)
- huggingface with an account to access datasets

```
conda create -n ai-text-detection python=3.12
conda activate ai-text-detection

conda install numpy pandas scikit-learn torch transformers
pip install -r requirements.txt

# then for CUDA 
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# if running on laptop without GPU, use CPU version
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu


```

## Usage
1. Log into hf using Run get_datasets.py to download datasets: `python get_datasets.py`.
2. Run feature extraction: `python extract_features.py --dataset <path> --output <path>. If no path, defaults are used. 
3. Train/evaluate: `python train.py --model xgboost --features all`.
4. Analyze: `python analyze_importance.py`.


1. Log into Hugging Face and download datasets: `python get_datasets.py`.
2. Configure dataset_configuration.toml for dataset combination (e.g., enable sources, set total samples, portions), then combine  datasets: `python combine_dataset.py`. 
3. Configure `configuration.toml` for model settings (e.g., num_samples, enabled features, voting, feature params), then train and evaluate using: `python main.py`.

## Contributors
- Finn Fonteijn
- Linn Gregussen

