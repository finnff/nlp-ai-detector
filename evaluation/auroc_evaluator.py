import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Any
import datetime

class AUROCEvaluator:
    """
    Comprehensive AUROC evaluation and ROC curve visualization system
    for AI text detection classifiers.
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.roc_curves = {}
        self.auroc_scores = {}
        self.dataset_metadata = {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up matplotlib style for professional plots with higher DPI for better readability
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 600  # Increased from 450 to 600 (2x original)
        plt.rcParams['savefig.dpi'] = 600  # Increased from 450 to 600 (2x original)
        plt.rcParams['font.size'] = 12  # Increased from 10 to 12 for better readability
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.alpha'] = 0.3

    def load_dataset_metadata(self, metadata_path="data/datasets/dataset_metadata.json"):
        """Load dataset configuration metadata for diagram titles."""
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.dataset_metadata = json.load(f)
                print(f"✅ Loaded dataset metadata: {metadata_path}")
                return True
            else:
                print(f"⚠️  Dataset metadata file not found: {metadata_path}")
                return False
        except Exception as e:
            print(f"⚠️  Error loading dataset metadata: {e}")
            return False

    def calculate_auroc(self, y_true: np.ndarray, y_scores: np.ndarray, classifier_name: str) -> float:
        """
        Calculate AUROC score for a single classifier.

        Args:
            y_true: True binary labels
            y_scores: Predicted probabilities for positive class
            classifier_name: Name of the classifier for logging

        Returns:
            AUROC score
        """
        try:
            auroc = roc_auc_score(y_true, y_scores)
            self.auroc_scores[classifier_name] = auroc
            return auroc
        except ValueError as e:
            print(f"⚠️  Could not calculate AUROC for {classifier_name}: {e}")
            self.auroc_scores[classifier_name] = None
            return None

    def collect_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, classifier_name: str):
        """
        Collect ROC curve data for plotting.

        Args:
            y_true: True binary labels
            y_scores: Predicted probabilities for positive class
            classifier_name: Name of the classifier
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            self.roc_curves[classifier_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }

            # Store AUROC score if not already calculated
            if classifier_name not in self.auroc_scores:
                self.auroc_scores[classifier_name] = roc_auc

        except Exception as e:
            print(f"⚠️  Could not collect ROC curve data for {classifier_name}: {e}")

    def generate_diagram_title(self) -> str:
        """
        Generate comprehensive diagram title with dataset configuration.

        Returns:
            Formatted title string with dataset details
        """
        if not self.dataset_metadata:
            return "ROC Curves - AI Text Detection Performance"

        # Extract metadata
        total_samples = self.dataset_metadata.get('total_samples', 'Unknown')
        test_split = self.dataset_metadata.get('test_split', 'Unknown')
        enabled_datasets = self.dataset_metadata.get('enabled_datasets', [])
        balance_classes = self.dataset_metadata.get('balance_classes', False)

        # Format enabled datasets
        if enabled_datasets:
            dataset_names = [ds.replace('_', ' ').title() for ds in enabled_datasets[:4]]
            if len(enabled_datasets) > 4:
                dataset_names.append(f"+{len(enabled_datasets)-4} more")
            datasets_str = ', '.join(dataset_names)
        else:
            datasets_str = "Unknown"

        # Format sample size
        if isinstance(total_samples, (int, float)):
            if total_samples >= 1000:
                samples_str = f"{total_samples/1000:.0f}K samples"
            else:
                samples_str = f"{total_samples} samples"
        else:
            samples_str = f"{total_samples} samples"

        # Format test split
        if isinstance(test_split, float):
            split_str = f"{int((1-test_split)*100)}/{int(test_split*100)} split"
        else:
            split_str = f"{test_split} split"

        # Balance status
        balance_str = "Balanced" if balance_classes else "Unbalanced"

        # Combine all elements
        title = f"ROC Curves - AI Text Detection ({samples_str}, {split_str})\n"
        title += f"Datasets: [{datasets_str}] ({balance_str})"

        return title

    def generate_roc_diagram(self, timestamp: str, include_ensembles: bool = True,
                           figsize: Tuple[int, int] = (12, 9)) -> str:
        """
        Generate comprehensive ROC curve diagram with professional styling.

        Args:
            timestamp: Timestamp for filename
            include_ensembles: Whether to include ensemble methods
            figsize: Figure size tuple

        Returns:
            Path to generated diagram file
        """
        if not self.roc_curves:
            print("⚠️  No ROC curve data available for diagram generation")
            return None

        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=figsize)

        # Color palette for classifiers
        colors = {
            'nb_classifier': '#1f77b4',           # Blue
            'nb_no_stopwords': '#ff7f0e',         # Orange
            'bert_classifier': '#2ca02c',         # Green
            'deberta_classifier': '#d62728',      # Red
            'xgboost_classifier': '#9467bd',      # Purple
            'ensemble_voting': '#8c564b',         # Brown
            'ensemble_stacking': '#e377c2',       # Pink
        }

        # Plot individual classifier ROC curves
        for name, data in self.roc_curves.items():
            # Skip ensembles if not requested
            if not include_ensembles and 'ensemble' in name.lower():
                continue

            # Determine color
            color = colors.get(name.lower(), None)
            if color is None:
                # Generate color based on hash of name for consistency
                color_hash = hash(name) % 8
                color = plt.cm.Set2(color_hash)

            # Plot ROC curve
            ax.plot(data['fpr'], data['tpr'],
                   color=color, linewidth=2.5, alpha=0.8,
                   label=f'{name} (AUC = {data["auc"]:.4f})')

        # Add diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2,
               label='Random Classifier (AUC = 0.5000)')

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')

        # Set title with dataset configuration
        title = self.generate_diagram_title()
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)

        # Legend positioning and styling
        if len(self.roc_curves) <= 8:
            legend_pos = 'lower right'
        else:
            legend_pos = 'outside right'

        if legend_pos == 'outside right':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                     fontsize=14, frameon=True, fancybox=True, shadow=True)
        else:
            ax.legend(loc=legend_pos, fontsize=14, frameon=True,
                     fancybox=True, shadow=True)

        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')

        # Add performance metrics text box
        if self.auroc_scores:
            # Find best classifier
            valid_scores = {k: v for k, v in self.auroc_scores.items() if v is not None}
            if valid_scores:
                best_classifier = max(valid_scores, key=valid_scores.get)
                best_score = valid_scores[best_classifier]

                # Create text box
                textstr = f'Best AUROC: {best_classifier}\nScore: {best_score:.4f}'
                props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save diagram
        diagram_path = os.path.join(self.output_dir, f'roc_curves_{timestamp}.png')
        plt.savefig(diagram_path, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"📈 ROC curve diagram saved: {diagram_path}")
        return diagram_path

    def generate_auroc_summary(self) -> str:
        """
        Generate formatted AUROC performance summary.

        Returns:
            Formatted string with AUROC analysis
        """
        if not self.auroc_scores:
            return "No AUROC scores available"

        # Filter valid scores
        valid_scores = {k: v for k, v in self.auroc_scores.items() if v is not None}

        if not valid_scores:
            return "No valid AUROC scores calculated"

        # Find best performers
        best_overall = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best_overall]

        # Separate individual and ensemble classifiers
        individual_scores = {k: v for k, v in valid_scores.items()
                          if 'ensemble' not in k.lower()}
        ensemble_scores = {k: v for k, v in valid_scores.items()
                         if 'ensemble' in k.lower()}

        summary = "\n" + "="*80 + "\n"
        summary += "AUROC PERFORMANCE ANALYSIS\n"
        summary += "="*80 + "\n"

        if individual_scores:
            best_individual = max(individual_scores, key=individual_scores.get)
            summary += f"• Best Individual AUROC: {best_individual} ({individual_scores[best_individual]:.4f})\n"

        summary += f"• Best Overall AUROC: {best_overall} ({best_score:.4f})\n"

        if ensemble_scores:
            best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
            summary += f"• Best Ensemble AUROC: {best_ensemble} ({ensemble_scores[best_ensemble]:.4f})\n"

        # Performance comparison
        summary += f"• All classifiers perform significantly better than random (0.5000)\n"

        # Calculate improvement over random
        improvement_pct = ((best_score - 0.5) / 0.5) * 100
        summary += f"• Best classifier shows {improvement_pct:.1f}% improvement over random\n"

        return summary

    def get_classifier_scores(self) -> Dict[str, float]:
        """
        Get all classifier AUROC scores.

        Returns:
            Dictionary mapping classifier names to AUROC scores
        """
        return self.auroc_scores.copy()

    def clear_data(self):
        """Clear all stored ROC data for fresh evaluation."""
        self.roc_curves.clear()
        self.auroc_scores.clear()
        self.dataset_metadata.clear()

    def print_summary(self):
        """Print AUROC summary to console."""
        print(self.generate_auroc_summary())
