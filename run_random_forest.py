import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

f, output_file = setup_results_file('random_forest')

X_train, X_test, y_train, y_test, feature_names = load_and_split_data_with_logging(
    args.features, test_size, random_state, f
)


print("Training Random Forest", f)

rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=random_state,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf, accuracy_rf = evaluate_model(rf_model, X_test, y_test, f)

print_feature_importance(
    feature_names,
    rf_model.feature_importances_,
    "Random Forest Feature Importance",
    f
)

f.close()
print(f"\nResults saved to {output_file}")