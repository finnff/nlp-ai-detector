from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import hashlib
import concurrent.futures
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any

class Ensemble:
    def __init__(self, classifiers, method='voting', stacking_estimator='LogisticRegression', disable_progress=False):
        """
        Ensemble class for combining multiple classifiers.

        Args:
            classifiers: Dict of {name: trained_classifier}
            method: 'voting' for majority vote, 'stacking' for meta-classifier
            stacking_estimator: Meta-classifier for stacking ('LogisticRegression', 'RandomForest', 'XGBoost')
            disable_progress: Whether to disable progress bars (default: False)
        """
        self.classifiers = classifiers
        self.method = method
        self.stacking_estimator = stacking_estimator
        self.disable_progress = disable_progress

        # Performance optimization: prediction caching
        self.prediction_cache = {}
        self.cached_predictions = {}  # Store all predictions for reuse

        # Track which methods are enabled
        if isinstance(method, str):
            self.enabled_methods = [method]
        else:
            self.enabled_methods = method

        # Initialize classifier type detection immediately to prevent AttributeError
        self._detect_classifier_types()

    def _detect_classifier_types(self):
        """Detect which classifiers need text vs features input."""
        self.text_classifiers = {}
        self.feature_classifiers = {}

        for name, clf in self.classifiers.items():
            # Check if classifier is XGBoost (uses features)
            if 'xgboost' in name.lower() or hasattr(clf, 'feature_importances_'):
                self.feature_classifiers[name] = clf
            else:
                # Assume all others use text
                self.text_classifiers[name] = clf

    def _get_data_hash(self, X_data) -> str:
        """Generate a hash for data to use as cache key."""
        if isinstance(X_data, list):
            data_str = str(len(X_data)) + str(X_data[0][:100] if X_data else "")
        elif hasattr(X_data, 'shape') and hasattr(X_data, 'dtype'):  # numpy array
            data_str = str(X_data.shape) + str(X_data.dtype) + str(X_data[0][:10] if X_data.size > 0 else "")
        elif hasattr(X_data, 'shape'):  # pandas DataFrame
            data_str = str(X_data.shape) + str(list(X_data.columns)[:5])  # Use columns instead of data[0]
        else:
            data_str = str(type(X_data)) + str(len(X_data) if hasattr(X_data, '__len__') else "0")

        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _get_cached_predictions(self, classifier_name: str, X_data, data_type: str) -> List[int]:
        """Get cached predictions or compute and cache them."""
        data_hash = self._get_data_hash(X_data)
        cache_key = f"{classifier_name}_{data_type}_{data_hash}"

        if cache_key not in self.prediction_cache:
            if classifier_name in self.text_classifiers:
                classifier = self.text_classifiers[classifier_name]
            else:
                classifier = self.feature_classifiers[classifier_name]

            predictions = classifier.predict(X_data)
            self.prediction_cache[cache_key] = predictions

        return self.prediction_cache[cache_key]

    def get_all_predictions(self, X_text, X_features=None) -> Dict[str, List[int]]:
        """Get all classifier predictions once and cache them for reuse."""
        # Check if we already have cached predictions for this data combination
        text_hash = self._get_data_hash(X_text)
        features_hash = self._get_data_hash(X_features) if X_features is not None else "no_features"
        cache_key = f"all_predictions_{text_hash}_{features_hash}"

        if cache_key in self.cached_predictions:
            return self.cached_predictions[cache_key]

        all_predictions = {}

        # Collect classifier tasks for parallel processing
        classifier_tasks = {}

        # Text classifiers
        for name in self.text_classifiers.keys():
            classifier_tasks[name] = (self.text_classifiers[name], X_text, "text")

        # Feature classifiers
        if X_features is not None:
            for name in self.feature_classifiers.keys():
                classifier_tasks[name] = (self.feature_classifiers[name], X_features, "features")

        # Process predictions in parallel
        if len(classifier_tasks) > 1 and not self.disable_progress:
            all_predictions = self._predict_parallel(classifier_tasks)
        else:
            # Sequential processing for single classifier or disabled progress
            for name, (clf, data, data_type) in classifier_tasks.items():
                all_predictions[name] = clf.predict(data)

        # Cache the results
        self.cached_predictions[cache_key] = all_predictions
        return all_predictions

    def _predict_parallel(self, classifier_tasks: Dict[str, Tuple]) -> Dict[str, List[int]]:
        """Run classifier predictions in parallel with enhanced progress tracking."""
        results = {}

        with tqdm(total=len(classifier_tasks), desc="Running classifiers in parallel",
                 disable=self.disable_progress, unit="classifier") as pbar:

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(classifier_tasks))) as executor:
                # Submit all prediction tasks
                future_to_name = {}
                for name, (clf, data, data_type) in classifier_tasks.items():
                    pbar.set_description(f"Submitting {name}")
                    # Use enhanced prediction method with progress
                    if hasattr(clf, 'predict_proba'):  # Most sklearn classifiers
                        future = executor.submit(self._predict_with_sample_progress, clf, data, name, data_type)
                    else:
                        future = executor.submit(clf.predict, data)
                    future_to_name[future] = name

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        pbar.set_description(f"Completed {name}")
                        results[name] = future.result()
                        pbar.update(1)

                        # Update progress bar with additional info
                        if results[name] and len(results[name]) > 0:
                            pbar.set_postfix({
                                "samples": len(results[name]),
                                "status": "✓"
                            })
                    except Exception as e:
                        print(f"⚠️  Prediction failed for {name}: {e}")
                        pbar.update(1)
                        pbar.set_postfix({
                            "samples": "N/A",
                            "status": "✗"
                        })

        return results

    def _build_meta_features(self, predictions: Dict[str, List[int]]) -> Any:
        """Build meta-features from cached predictions efficiently."""
        import numpy as np

        meta_features = []
        for name in self.meta_feature_names:
            if name in predictions:
                pred_array = np.array(predictions[name]).reshape(-1, 1)
                meta_features.append(pred_array)

        if len(meta_features) > 1:
            return np.hstack(meta_features)
        elif len(meta_features) == 1:
            return meta_features[0]
        else:
            raise ValueError("No valid predictions for meta-feature construction")

    def predict_all_methods(self, X_text, X_features=None) -> Dict[str, List[int]]:
        """Get predictions for all enabled methods (voting, stacking) in one optimized call."""
        all_predictions = self.get_all_predictions(X_text, X_features)
        results = {}

        # Voting results
        if 'voting' in self.enabled_methods:
            results['voting'] = self._majority_vote(all_predictions)

        # Stacking results
        if 'stacking' in self.enabled_methods:
            if hasattr(self, 'meta_classifier'):
                try:
                    X_meta = self._build_meta_features(all_predictions)
                    results['stacking'] = self.meta_classifier.predict(X_meta)
                except Exception as e:
                    print(f"⚠️  Stacking prediction failed: {e}, falling back to voting")
                    results['stacking'] = self._majority_vote(all_predictions)  # Fallback
            else:
                print(f"⚠️  Stacking not available (no meta-classifier), falling back to voting")
                results['stacking'] = self._majority_vote(all_predictions)  # Fallback

        return results

    def _majority_vote(self, predictions: Dict[str, List[int]]) -> List[int]:
        """Perform majority voting on cached predictions."""
        if not predictions:
            raise ValueError("No predictions available for voting")

        num_samples = len(next(iter(predictions.values())))
        y_pred = []

        for i in range(num_samples):
            votes = [predictions[name][i] for name in predictions]
            majority = max(set(votes), key=votes.count)
            y_pred.append(majority)

        return y_pred

    def fit(self, X_text, y, X_features=None):
        """
        Fit the ensemble.

        Args:
            X_text: Text data for text-based classifiers
            y: Labels
            X_features: Feature data for feature-based classifiers (optional)
        """
        self._detect_classifier_types()

        # Check if stacking should be processed (handle multiple methods)
        stacking_enabled = 'stacking' in self.enabled_methods

        if stacking_enabled:
            # For stacking with mixed classifier types, we'll use a manual approach
            # that doesn't rely on sklearn's StackingClassifier (which has compatibility issues)
            print(f"🔧 Creating stacking ensemble with {len(self.text_classifiers) + len(self.feature_classifiers)} classifiers")
            print(f"   Text: {list(self.text_classifiers.keys())}, Features: {list(self.feature_classifiers.keys())}")

            # Get predictions from all classifiers to use as meta-features
            meta_features = []
            total_classifiers = len(self.text_classifiers) + (len(self.feature_classifiers) if X_features is not None else 0)

            print(f"🔧 Collecting meta-features from {total_classifiers} classifiers...")

            # Get predictions from text classifiers with enhanced progress
            for name, clf in self.text_classifiers.items():
                try:
                    pred = self._predict_with_sample_progress(clf, X_text, name, "text")
                    import numpy as np
                    pred_array = np.array(pred).reshape(-1, 1)
                    meta_features.append(pred_array)
                except Exception as e:
                    print(f"⚠️  Could not get predictions from {name}: {e}")

            # Get predictions from feature classifiers with enhanced progress
            if X_features is not None:
                for name, clf in self.feature_classifiers.items():
                    try:
                        pred = self._predict_with_sample_progress(clf, X_features, name, "features")
                        import numpy as np
                        pred_array = np.array(pred).reshape(-1, 1)
                        meta_features.append(pred_array)
                    except Exception as e:
                        print(f"⚠️  Could not get predictions from {name}: {e}")

            print(f"✅ Collected {len(meta_features)} meta-features from {total_classifiers} classifiers")

            if len(meta_features) > 1:
                # Combine all predictions as meta-features
                import numpy as np
                X_meta = np.hstack(meta_features)
                print(f"🔧 Combined meta-features: {X_meta.shape[0]} samples, {X_meta.shape[1]} features")

                # Train meta-classifier
                estimator_map = {
                    "LogisticRegression": lambda: LogisticRegression(),
                    "RandomForest": lambda: RandomForestClassifier(),
                    "XGBoost": lambda: __import__('xgboost').XGBClassifier()
                }
                stacking_estimator = getattr(self, 'stacking_estimator', 'LogisticRegression')

                try:
                    final_estimator = estimator_map[stacking_estimator]()

                    # Use enhanced meta-classifier training with progress
                    self.meta_classifier = self._train_meta_classifier_with_progress(
                        final_estimator, X_meta, y, stacking_estimator
                    )
                    self.meta_feature_names = list(self.text_classifiers.keys()) + list(self.feature_classifiers.keys())
                    print(f"✅ Successfully trained {stacking_estimator} meta-classifier")

                except KeyError:
                    print(f"Warning: Unknown stacking_estimator '{stacking_estimator}', using LogisticRegression")
                    fallback_estimator = LogisticRegression()
                    self.meta_classifier = self._train_meta_classifier_with_progress(
                        fallback_estimator, X_meta, y, "LogisticRegression"
                    )
                    self.meta_feature_names = list(self.text_classifiers.keys()) + list(self.feature_classifiers.keys())
                    print(f"✅ Successfully trained fallback LogisticRegression meta-classifier")

                except Exception as e:
                    print(f"❌ Error training meta-classifier ({e}), falling back to voting")
                    print(f"   Exception type: {type(e).__name__}")
                    # Don't change self.method, just report the error
            else:
                print(f"❌ Warning: Not enough valid classifiers for stacking (need >1, got {len(meta_features)})")
                # Don't change self.method, let predict_all_methods handle the fallback

    def predict(self, X_text, X_features=None):
        """
        Predict using the ensemble method (optimized version).

        Args:
            X_text: Text data for text-based classifiers
            X_features: Feature data for feature-based classifiers (optional)

        Returns:
            List of predictions
        """
        if self.method == 'voting':
            # Use optimized approach: get all predictions once, cache, then vote
            all_predictions = self.get_all_predictions(X_text, X_features)
            return self._majority_vote(all_predictions)
        elif self.method == 'stacking':
            if hasattr(self, 'meta_classifier'):
                # Use optimized stacking approach: get cached predictions, build meta-features, predict
                try:
                    all_predictions = self.get_all_predictions(X_text, X_features)
                    X_meta = self._build_meta_features(all_predictions)

                    # Use enhanced progress for meta-classifier prediction
                    result = self._predict_with_sample_progress(
                        self.meta_classifier, X_meta, "Meta-classifier", "meta-features"
                    )
                    return result
                except Exception as e:
                    print(f"Warning: Stacking prediction failed ({e}), falling back to voting")
                    self.method = 'voting'
                    return self.predict(X_text, X_features)
            else:
                # Fallback to voting if stacking wasn't set up
                print("Warning: Stacking not available, falling back to voting")
                self.method = 'voting'
                return self.predict(X_text, X_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _predict_with_sample_progress(self, clf, X_data, classifier_name, data_type="text"):
        """
        Show sample-wise progress during prediction instead of classifier-wise.

        Args:
            clf: The classifier object
            X_data: Input data (list or array)
            classifier_name: Name for progress display
            data_type: Type of data being processed

        Returns:
            List of predictions
        """
        try:
            total_samples = len(X_data)

            # For sklearn models that can handle batch processing efficiently,
            # still show progress but process in reasonable batch sizes
            if hasattr(clf, 'predict_proba'):  # Most sklearn classifiers
                batch_size = min(1000, total_samples)
                predictions = []

                with tqdm(total=total_samples,
                         desc=f"Processing {classifier_name} ({data_type})",
                         unit="samples",
                         disable=self.disable_progress) as pbar:

                    start_time = time.time()
                    for i in range(0, total_samples, batch_size):
                        batch_end = min(i + batch_size, total_samples)
                        batch = X_data[i:batch_end]

                        # Process batch
                        batch_pred = clf.predict(batch)
                        predictions.extend(batch_pred)

                        # Update progress with metrics
                        pbar.update(len(batch))
                        if pbar.last_iter_t > 0:  # Avoid division by zero
                            samples_per_sec = len(batch) / pbar.last_iter_t
                            elapsed = time.time() - start_time
                            eta = (total_samples - pbar.n) / samples_per_sec if samples_per_sec > 0 else 0
                            pbar.set_postfix({
                                "samples/s": f"{samples_per_sec:.0f}",
                                "ETA": f"{eta:.1f}s"
                            })

                return predictions
            else:
                # For other classifiers, show single progress bar but with meaningful description
                with tqdm(total=1, desc=f"Running {classifier_name} ({data_type})",
                         unit="prediction", disable=self.disable_progress) as pbar:
                    start_time = time.time()
                    predictions = clf.predict(X_data)
                    elapsed = time.time() - start_time
                    pbar.set_postfix({
                        "samples": total_samples,
                        "time": f"{elapsed:.2f}s",
                        "samples/s": f"{total_samples/elapsed:.0f}" if elapsed > 0 else "N/A"
                    })
                    pbar.update(1)

                return predictions

        except Exception as e:
            print(f"⚠️  Progress tracking failed for {classifier_name}, falling back to normal prediction: {e}")
            return clf.predict(X_data)

    def _train_meta_classifier_with_progress(self, estimator, X_meta, y, estimator_name):
        """
        Show enhanced progress during meta-classifier training.

        Args:
            estimator: The meta-classifier estimator
            X_meta: Meta-features
            y: Target labels
            estimator_name: Name for progress display

        Returns:
            Trained estimator
        """
        try:
            n_samples = len(X_meta)

            # For LogisticRegression and similar iterative models
            if hasattr(estimator, 'max_iter') and hasattr(estimator, 'n_iter_'):
                with tqdm(total=estimator.max_iter,
                         desc=f"Training {estimator_name}",
                         unit="iter",
                         disable=self.disable_progress) as pbar:

                    # Wrap the fit method to show iteration progress
                    original_fit = estimator.fit

                    def fit_with_progress(X, y_sample):
                        result = original_fit(X, y_sample)
                        if hasattr(estimator, 'n_iter_'):
                            # Update progress to show actual iterations completed
                            pbar.n = estimator.n_iter_[0]  # n_iter_ is usually an array
                            pbar.set_description(f"Training {estimator_name} (converged at iter {estimator.n_iter_[0]})")
                            pbar.refresh()
                        return result

                    estimator.fit = fit_with_progress
                    start_time = time.time()
                    result = estimator.fit(X_meta, y)
                    elapsed = time.time() - start_time
                    estimator.fit = original_fit

                    # Show final training stats
                    if hasattr(estimator, 'n_iter_'):
                        pbar.set_postfix({
                            "converged": True,
                            "time": f"{elapsed:.2f}s",
                            "samples": n_samples
                        })
                    else:
                        pbar.set_postfix({
                            "time": f"{elapsed:.2f}s",
                            "samples": n_samples
                        })

                    return result

            else:
                # For non-iterative models, show sample-wise progress
                with tqdm(total=n_samples,
                         desc=f"Training {estimator_name}",
                         unit="samples",
                         disable=self.disable_progress) as pbar:

                    start_time = time.time()
                    result = estimator.fit(X_meta, y)
                    elapsed = time.time() - start_time

                    pbar.update(n_samples)
                    pbar.set_postfix({
                        "time": f"{elapsed:.2f}s",
                        "samples/s": f"{n_samples/elapsed:.0f}" if elapsed > 0 else "N/A"
                    })

                    return result

        except Exception as e:
            print(f"⚠️  Enhanced progress tracking failed for {estimator_name}, using standard training: {e}")
            return estimator.fit(X_meta, y)

    def evaluate(self, y_test, y_pred, target_names=['Human Written', 'AI Generated']):
        """
        Evaluate the ensemble predictions.
        """
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        return accuracy, report