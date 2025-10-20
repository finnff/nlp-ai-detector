import pandas as pd
import xgboost as xgb
from linguistic_feature_model_utils import (
    setup_argparser,
    load_config,
    load_and_split_data_with_logging,
    setup_results_file,
    dual_print,
    evaluate_model,
    print_feature_importance
)

args = setup_argparser()
config = load_config(args.config)

test_size = config['test_size']
random_state = config['random_state']

f, output_file = setup_results_file('xgboost')

X_train, X_test, y_train, y_test, feature_names = load_and_split_data_with_logging(
    args.features, test_size, random_state, f
)

print("\nTraining XGBoost", f)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=random_state,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_pred, accuracy = evaluate_model(model, X_test, y_test, f)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print_feature_importance(
    feature_names,
    model.feature_importances_,
    "Feature Importances",
    f
)

f.close()
print(f"\nResults saved to {output_file}")

