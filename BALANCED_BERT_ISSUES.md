

# When training on 172k unbalanced samples (64% human) versus 71k balanced samples (~50% each), the unbalanced dataset yields way better accuracy (BERT 0.9190 vs. 0.8316)

Balancing the dataset reduces overall accuracy (e.g., BERT drops from 0.9190 to 0.8316, ensemble from 0.8783 to 0.8710)
primarily because it limits the total samples by discarding excess majority-class data per source (e.g., from 172K to 71K
samples). This data loss hurts model training, especially for data-hungry models like BERT, leading to worse
generalization.

You're not missing anything in F1/recall— the unbalanced version has better macro F1 (0.91 vs. 0.83) and more balanced
recalls (human 0.92 vs. 0.68, AI 0.91 vs. 0.98), as the larger dataset allows better learning without severe bias issues.
The imbalance (64% human) isn't extreme enough to dominate performance, and the extra data outweighs the class skew.
Balancing helps fairness but can underperform here due to sample reduction; consider oversampling or weighted loss instead.

# Adjusting for dataset size variation using a 72k unbalanced subset (61.78% human) versus the 71k balanced set (~50% each), the unbalanced data still outperforms (BERT 0.8941 vs. 0.8316, macro F1 0.89 vs. 0.83).


The unbalanced dataset (72,958 samples, 61.78% human) outperforms the balanced one (71,586 samples, ~50% each) even at
similar sizes, with BERT accuracy 0.8941 vs. 0.8316 and macro F1 0.89 vs. 0.83. This is because balancing discards excess
majority-class samples (often humans from imbalanced sources like llm_detect_competition), reducing data diversity and
training signal for the harder-to-classify human class. The unbalanced version retains more human examples, improving
recall for humans (0.95 vs. 0.68) and overall metrics, as the model benefits from the natural distribution without severe
bias issues. For this task, imbalance aids performance; if balance is needed, try weighted loss or SMOTE instead of
undersampling.


Maybe a reason not to use balancing is that the human-written text is more diverse and harder to learn than the AI-generated text, so having more human-written samples helps the model learn better decision boundaries. In contrast, the AI-generated text might be more formulaic or similar across samples, so fewer examples are needed to capture its characteristics. Thus, reducing human samples to balance the classes could disproportionately hurt performance by limiting exposure to the more complex class?
