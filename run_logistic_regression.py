import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from linguistic_feature_model_utils import (
    setup_argparser,
    load_config,
    load_and_split_data_with_logging,
    setup_results_file,
    dual_print,
    evaluate_model
)

args = setup_argparser()
config = load_config(args.config)

test_size = config['test_size']
random_state = config['random_state']

f, output_file = setup_results_file('logistic_regression')

X_train, X_test, y_train, y_test, feature_names = load_and_split_data_with_logging(
    args.features, test_size, random_state, f
)

print("Training Logistic Regression", f)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr, accuracy_lr = evaluate_model(lr_model, X_test_scaled, y_test, f)

# normalize, since logistic regression gives coefficients, while random forest and zgboost already normalize. 
abs_coef = np.abs(lr_model.coef_[0])
normalized_importance = abs_coef / abs_coef.sum()

lr_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': abs_coef,
    'normalized_importance': normalized_importance
}).sort_values('abs_coefficient', ascending=False)

dual_print("\nLogistic Regression Feature Importance:", f)
dual_print(lr_importance, f)

f.close()
print(f"\nResults saved to {output_file}")
