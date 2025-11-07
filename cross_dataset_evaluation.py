#!/usr/bin/env python3
"""
Cross-Dataset Evaluation for XGBoost AI Text Detection Classifier

This script performs leave-one-dataset-out cross-validation to evaluate
how well the XGBoost classifier generalizes to unseen datasets.

Usage:
    python cross_dataset_evaluation.py

Output:
    - results/cross_dataset_evaluation.json (detailed metrics)
    - results/cross_dataset_summary.txt (human-readable summary)
"""

import os
import sys
import json
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.xgboost_classifier_with_save import XGBoostClassifierWithSave
from evaluation.auroc_evaluator import AUROCEvaluator
from combine_dataset import create_training_dataset, load_individual_dataset

class CrossDatasetResultsTracker:
    """Track and aggregate cross-dataset evaluation results"""

    def __init__(self):
        self.results = {}
        self.summary_stats = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_result(self, test_dataset: str, metrics: Dict[str, Any]):
        """Add results for a specific test dataset"""
        self.results[test_dataset] = {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'auroc': metrics['auroc'],
            'classification_report': metrics['classification_report'],
            'test_samples': len(metrics['predictions']) if 'predictions' in metrics else None
        }

    def calculate_summary_stats(self):
        """Calculate overall statistics across all test datasets"""
        if not self.results:
            return

        accuracies = [r['accuracy'] for r in self.results.values()]
        f1_macros = [r['f1_macro'] for r in self.results.values()]
        f1_weighteds = [r['f1_weighted'] for r in self.results.values()]
        aurocs = [r['auroc'] for r in self.results.values() if r['auroc'] is not None]

        self.summary_stats = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1_macro': np.mean(f1_macros),
            'std_f1_macro': np.std(f1_macros),
            'mean_f1_weighted': np.mean(f1_weighteds),
            'std_f1_weighted': np.std(f1_weighteds),
            'mean_auroc': np.mean(aurocs) if aurocs else None,
            'std_auroc': np.std(aurocs) if aurocs else None,
            'num_experiments': len(self.results)
        }

    def save_results(self, output_dir: str = "results"):
        """Save detailed results and summary to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Calculate summary stats before saving
        self.calculate_summary_stats()

        # Save detailed JSON results
        json_results = {
            'timestamp': self.timestamp,
            'experiment_type': 'cross_dataset_evaluation',
            'model': 'xgboost_classifier',
            'individual_results': self.results,
            'summary_statistics': self.summary_stats
        }

        json_path = os.path.join(output_dir, f"cross_dataset_evaluation_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save human-readable summary
        summary_path = os.path.join(output_dir, f"cross_dataset_summary_{self.timestamp}.txt")
        self._save_summary_txt(summary_path)

        print(f"✅ Results saved:")
        print(f"   📊 Detailed: {json_path}")
        print(f"   📋 Summary: {summary_path}")

        return json_path, summary_path

    def _save_summary_txt(self, summary_path: str):
        """Save human-readable summary text file"""
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CROSS-DATASET EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Model: XGBoost Classifier\n")
            f.write(f"Experiment Type: Leave-One-Dataset-Out Cross-Validation\n")
            f.write(f"Number of Experiments: {len(self.results)}\n\n")

            # Individual results
            f.write("INDIVIDUAL DATASET RESULTS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Test Dataset':<15} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'AUROC':<8}\n")
            f.write("-" * 60 + "\n")

            for dataset, result in self.results.items():
                acc = result['accuracy']
                f1_macro = result['f1_macro']
                f1_weighted = result['f1_weighted']
                auroc = result['auroc']
                auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"

                f.write(f"{dataset:<15} {acc:<10.4f} {f1_macro:<10.4f} {f1_weighted:<12.4f} {auroc_str:<8}\n")

            f.write("\n")

            # Summary statistics
            if self.summary_stats:
                f.write("SUMMARY STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Accuracy:     {self.summary_stats['mean_accuracy']:.4f} ± {self.summary_stats['std_accuracy']:.4f}\n")
                f.write(f"Mean Macro F1:     {self.summary_stats['mean_f1_macro']:.4f} ± {self.summary_stats['std_f1_macro']:.4f}\n")
                f.write(f"Mean Weighted F1:  {self.summary_stats['mean_f1_weighted']:.4f} ± {self.summary_stats['std_f1_weighted']:.4f}\n")
                if self.summary_stats['mean_auroc'] is not None:
                    f.write(f"Mean AUROC:        {self.summary_stats['mean_auroc']:.4f} ± {self.summary_stats['std_auroc']:.4f}\n")
                f.write("\n")

            # Best and worst performing datasets
            if self.results:
                f.write("PERFORMANCE ANALYSIS:\n")
                f.write("-" * 40 + "\n")

                best_acc = max(self.results.items(), key=lambda x: x[1]['accuracy'])
                worst_acc = min(self.results.items(), key=lambda x: x[1]['accuracy'])

                f.write(f"Best Accuracy:  {best_acc[0]} ({best_acc[1]['accuracy']:.4f})\n")
                f.write(f"Worst Accuracy: {worst_acc[0]} ({worst_acc[1]['accuracy']:.4f})\n")

                if any(r['auroc'] is not None for r in self.results.values()):
                    auroc_results = [(k, v) for k, v in self.results.items() if v['auroc'] is not None]
                    if auroc_results:
                        best_auroc = max(auroc_results, key=lambda x: x[1]['auroc'])
                        worst_auroc = min(auroc_results, key=lambda x: x[1]['auroc'])
                        f.write(f"Best AUROC:    {best_auroc[0]} ({best_auroc[1]['auroc']:.4f})\n")
                        f.write(f"Worst AUROC:   {worst_auroc[0]} ({worst_auroc[1]['auroc']:.4f})\n")


class CrossDatasetEvaluator:
    """Main cross-dataset evaluation orchestrator"""

    def __init__(self, config_path: str = "dataset_configuration.toml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.datasets_to_test = ['hc3', 'daigt_v2', 'sunilthite', 'ah_aitd']
        self.results_tracker = CrossDatasetResultsTracker()
        self.output_dir = "results"

    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, 'rb') as f:
            return tomllib.load(f)

    def _create_training_config(self, exclude_dataset: str) -> Dict[str, Any]:
        """Create configuration with specified dataset excluded"""
        training_config = self.config.copy()

        # Disable the excluded dataset
        if exclude_dataset in training_config:
            training_config[exclude_dataset]['enabled'] = False

        return training_config

    def _get_enabled_datasets(self, config: Dict[str, Any]) -> List[str]:
        """Get list of enabled datasets from configuration"""
        enabled = []
        for key, value in config.items():
            if key not in ['perplexity', 'Title', 'total_samples', 'use_arrow', 'random', 'balance_classes']:
                if isinstance(value, dict) and value.get('enabled', False):
                    enabled.append(key)
        return enabled

    def _load_individual_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset for testing"""
        return load_individual_dataset(dataset_name)

    def _create_training_dataset(self, exclude_dataset: str) -> pd.DataFrame:
        """Create training dataset with specified dataset excluded"""
        return create_training_dataset(exclude_dataset, self.config_path)

    def _extract_features(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Extract linguistic features and return path to features file"""
        print(f"Extracting features for {dataset_name}...")

        # Create temporary dataset for feature extraction
        temp_dir = "temp_cross_dataset"
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, f"{dataset_name}_temp.csv")
        df.to_csv(temp_path, index=False)

        # Extract features using existing utility
        features_path = os.path.join(temp_dir, f"{dataset_name}_features.csv")

        # Save original sys.argv and replace with arguments for feature extraction
        original_argv = sys.argv
        sys.argv = ['extract_linguistic_features.py', '--dataset', temp_path, '--output', features_path]

        try:
            import extract_linguistic_features as extract_features
            extract_features.main()
            print(f"✅ Features extracted to {features_path}")
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return None
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

        return features_path

    def _train_xgboost(self, features_path: str) -> XGBoostClassifierWithSave:
        """Train XGBoost classifier on extracted features"""
        print("Training XGBoost classifier...")

        # Initialize classifier with existing features file
        classifier = XGBoostClassifierWithSave(
            features_file=features_path,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            test_size=0.2  # This won't be used since we provide training data
        )

        # Load features and train
        X, y = classifier.load_features()

        # Use all data for training (no test split for cross-dataset eval)
        classifier.fit(X, y, scale_features=True)

        print("✅ XGBoost training completed")
        return classifier

    def _evaluate_on_dataset(self, classifier: XGBoostClassifierWithSave,
                           test_features_path: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate trained classifier on test dataset"""
        print(f"Evaluating on {dataset_name}...")

        # Load test features
        test_df = pd.read_csv(test_features_path)
        X_test = test_df.drop('generated', axis=1)
        y_test = test_df['generated']

        # Evaluate
        results = classifier.evaluate(X_test, y_test)

        print(f"✅ Evaluation on {dataset_name} completed")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Macro F1: {results['f1_macro']:.4f}")
        print(f"   Weighted F1: {results['f1_weighted']:.4f}")
        if results['auroc'] is not None:
            print(f"   AUROC: {results['auroc']:.4f}")

        return results

    def run_evaluation(self):
        """Run complete cross-dataset evaluation"""
        print("=" * 80)
        print("CROSS-DATASET EVALUATION FOR XGBOOST CLASSIFIER")
        print("=" * 80)
        print(f"Datasets to test: {', '.join(self.datasets_to_test)}")
        print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        for test_dataset in self.datasets_to_test:
            print(f"\n{'='*20} Experiment {self.datasets_to_test.index(test_dataset) + 1}/4 {'='*20}")
            print(f"Test Dataset: {test_dataset}")
            print(f"Training Datasets: {[d for d in self.datasets_to_test if d != test_dataset]}")
            print()

            try:
                # Step 1: Create training dataset (exclude test dataset)
                train_df = self._create_training_dataset(test_dataset)
                if train_df.empty:
                    print(f"⚠️  Skipping {test_dataset} - no training data available")
                    continue

                # Step 2: Extract features from training data
                train_features_path = self._extract_features(train_df, f"train_excluding_{test_dataset}")

                # Step 3: Train XGBoost on training features
                classifier = self._train_xgboost(train_features_path)

                # Step 4: Load test dataset
                test_df = self._load_individual_dataset(test_dataset)
                if test_df.empty:
                    print(f"⚠️  Skipping {test_dataset} - no test data available")
                    continue

                # Step 5: Extract features from test data
                test_features_path = self._extract_features(test_df, f"test_{test_dataset}")

                # Step 6: Evaluate on test dataset
                results = self._evaluate_on_dataset(classifier, test_features_path, test_dataset)

                # Step 7: Store results
                self.results_tracker.add_result(test_dataset, results)

                print(f"✅ Experiment {self.datasets_to_test.index(test_dataset) + 1}/4 completed")

            except Exception as e:
                print(f"❌ Error processing {test_dataset}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save all results
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED - SAVING RESULTS")
        print("=" * 80)

        json_path, summary_path = self.results_tracker.save_results(self.output_dir)

        # Clean up temporary files
        self._cleanup_temp_files()

        return json_path, summary_path

    def _cleanup_temp_files(self):
        """Clean up temporary feature files"""
        import shutil
        temp_dir = "temp_cross_dataset"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary files from {temp_dir}")
            except Exception as e:
                print(f"⚠️  Could not clean up temporary files: {e}")


def main():
    """Main execution function"""
    print("🚀 Starting Cross-Dataset Evaluation for XGBoost Classifier")

    try:
        # Initialize evaluator
        evaluator = CrossDatasetEvaluator()

        # Run evaluation
        json_path, summary_path = evaluator.run_evaluation()

        print(f"\n🎉 Cross-dataset evaluation completed successfully!")
        print(f"📊 Check {summary_path} for human-readable results")
        print(f"📈 Check {json_path} for detailed metrics")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()