import torch
import numpy as np
import os
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from modules.utils.visualize import plot_confusion_matrix, plot_class_distribution
import seaborn as sns

class Tester:
    def __init__(self, model, test_loader, device, idx_to_chord=None, logger=None,
                 normalization=None, output_dir="results"):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.idx_to_chord = idx_to_chord
        self.logger = logger
        self.normalization = normalization
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _log(self, message):
        """Helper function to log messages."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def evaluate(self, save_plots=False, use_per_frame=True):
        """
        Evaluate the model on the test set.

        Args:
            save_plots: Whether to save evaluation plots
            use_per_frame: Whether to use per-frame prediction

        Returns:
            Dictionary of evaluation metrics
        """
        if self.test_loader is None:
            self._log("No test loader provided, skipping evaluation")
            return {}

        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []

        total_correct = 0
        total_samples = 0

        # Create a confusion matrix
        if self.idx_to_chord:
            n_classes = len(self.idx_to_chord)
        else:
            # Use a reasonable default if we don't know the exact number
            n_classes = 25

        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Process both dictionary and tuple format
                if isinstance(batch, dict):
                    inputs = batch['spectro']
                    targets = batch['chord_idx']
                else:
                    inputs, targets = batch

                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Apply normalization if provided
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']

                # Get predictions
                try:
                    # Check if model is wrapped with DistributedDataParallel
                    if hasattr(self.model, 'module'):
                        # Access the underlying model
                        base_model = self.model.module

                        # Try to use per-frame prediction methods on the base model
                        if hasattr(base_model, 'predict_per_frame') and use_per_frame:
                            preds = base_model.predict_per_frame(inputs)
                        elif hasattr(base_model, 'predict_frames') and use_per_frame:
                            preds = base_model.predict_frames(inputs)
                        elif hasattr(base_model, 'predict'):
                            preds = base_model.predict(inputs)
                        else:
                            # Fall back to direct model call
                            raise AttributeError("Base model has no predict method")
                    else:
                        # Regular model (not DDP wrapped)
                        if hasattr(self.model, 'predict_per_frame') and use_per_frame:
                            # Check if we should use the predict_per_frame method
                            preds = self.model.predict_per_frame(inputs)
                        elif hasattr(self.model, 'predict_frames') and use_per_frame:
                            # Alternative method name for per-frame prediction
                            preds = self.model.predict_frames(inputs)
                        elif hasattr(self.model, 'predict'):
                            # Fall back to standard predict method without per_frame parameter
                            preds = self.model.predict(inputs)
                        else:
                            # Fall back to direct model call
                            raise AttributeError("Model has no predict method")
                except Exception as e:
                    # Log the error but don't show it for every batch to reduce verbosity
                    if batch_idx == 0:
                        self._log(f"Error in model prediction: {e}")
                        self._log("Falling back to direct model call")

                    # Fall back to direct model call if predict method fails
                    outputs = self.model(inputs)

                    # Handle different model output formats
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Get predictions from logits
                    if logits.dim() == 3:  # [batch, time, classes]
                        preds = torch.argmax(logits, dim=2)
                    else:
                        preds = torch.argmax(logits, dim=-1)

                # Flatten tensors if needed for evaluation
                if preds.dim() > 1 and targets.dim() > 1:
                    # Both are sequences - flatten both
                    preds_flat = preds.view(-1)
                    targets_flat = targets.view(-1)
                elif preds.dim() > 1 and targets.dim() == 1:
                    # Predictions are sequences but targets are flat
                    # Take the mean prediction across the sequence
                    preds_flat = torch.mode(preds, dim=-1)[0]
                else:
                    # Both are already flat
                    preds_flat = preds
                    targets_flat = targets

                # Move to CPU for sklearn metrics
                preds_np = preds_flat.cpu().numpy()
                targets_np = targets_flat.cpu().numpy()

                # Accumulate predictions for metrics
                all_preds.append(preds_np)
                all_targets.append(targets_np)

                # Calculate accuracy
                correct = (preds_flat == targets_flat).sum().item()
                samples = targets_flat.size(0)

                total_correct += correct
                total_samples += samples

                # Update confusion matrix
                for t, p in zip(targets_np, preds_np):
                    # Ensure indices are within valid range
                    t_idx = min(t, n_classes - 1)
                    p_idx = min(p, n_classes - 1)
                    confusion_matrix[t_idx, p_idx] += 1

                # Log progress for long test sets
                if batch_idx % 10 == 0:
                    self._log(f"Evaluated {batch_idx} batches, current accuracy: {correct/max(1, samples):.4f}")

        # Calculate overall metrics
        accuracy = total_correct / max(1, total_samples)

        # Combine all predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate metrics with sklearn
        metrics = {
            'accuracy': accuracy,
            'samples': total_samples
        }

        # Calculate additional metrics if there are enough samples
        if len(all_targets) > 1:
            # Compute and add more detailed metrics
            try:
                metrics['precision_macro'] = precision_score(all_targets, all_preds, average='macro')
                metrics['recall_macro'] = recall_score(all_targets, all_preds, average='macro')
                metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro')

                # Calculate metrics for most frequent classes (top 5)
                class_counts = Counter(all_targets)
                top_classes = [cls for cls, _ in class_counts.most_common(5)]

                if len(top_classes) > 0:
                    # Create mask for top classes
                    mask = np.isin(all_targets, top_classes)
                    top_targets = all_targets[mask]
                    top_preds = all_preds[mask]

                    if len(top_targets) > 0:
                        metrics['top_accuracy'] = accuracy_score(top_targets, top_preds)
                        metrics['top_f1'] = f1_score(top_targets, top_preds, average='macro')

                        # Log top class names if mapping is available
                        if self.idx_to_chord:
                            top_class_names = [self.idx_to_chord.get(cls, f"Unknown-{cls}") for cls in top_classes]
                            self._log(f"Top classes: {top_class_names}")
            except Exception as e:
                self._log(f"Error calculating detailed metrics: {e}")

        # Save visualization if requested
        if save_plots:
            self._save_confusion_matrix(all_targets, all_preds)

        # Log results
        self._log(f"Test Results:")
        self._log(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'precision_macro' in metrics:
            self._log(f"Precision (macro): {metrics['precision_macro']:.4f}")
            self._log(f"Recall (macro): {metrics['recall_macro']:.4f}")
            self._log(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        if 'top_accuracy' in metrics:
            self._log(f"Top Classes Accuracy: {metrics['top_accuracy']:.4f}")
            self._log(f"Top Classes F1: {metrics['top_f1']:.4f}")

        return metrics

    def _analyze_confusion_matrix(self, targets, predictions, target_counter):
        """Analyze and log details from confusion matrix for common classes."""
        if len(target_counter) <= 1:
            self._log("Not enough classes to analyze confusion matrix")
            return

        # Get top classes by frequency
        top_classes = [idx for idx, _ in target_counter.most_common(10)]

        # Log confusion matrix for most common chords
        self._log("\nConfusion Matrix Analysis (Top Classes):")
        self._log(f"{'True Class':<20} | {'Accuracy':<10} | {'Most Predicted':<20} | {'Correct/Total'}")
        self._log(f"{'-'*20} | {'-'*10} | {'-'*20} | {'-'*15}")

        for true_idx in top_classes:
            true_chord = self.idx_to_chord.get(true_idx, str(true_idx)) if self.idx_to_chord else str(true_idx)

            # Find samples with this true class
            true_mask = (targets == true_idx)
            true_count = np.sum(true_mask)

            if true_count > 0:
                # Get predictions for these samples
                class_preds = predictions[true_mask]
                pred_counter = Counter(class_preds)

                # Calculate accuracy for this class
                correct = pred_counter.get(true_idx, 0)
                accuracy = correct / true_count

                # Get most common prediction for this class
                most_common_pred, most_common_count = pred_counter.most_common(1)[0]
                most_common_pred_chord = self.idx_to_chord.get(most_common_pred, str(most_common_pred)) if self.idx_to_chord else str(most_common_pred)

                self._log(f"{true_chord:<20} | {accuracy:.4f}     | {most_common_pred_chord:<20} | {correct}/{true_count}")

    def _generate_plots(self, targets, predictions, target_counter, pred_counter):
        """Generate and save visualization plots."""
        try:
            # 1. Class distribution plot
            if self.idx_to_chord:
                class_names = self.idx_to_chord
            else:
                class_names = None

            fig = plot_class_distribution(target_counter, class_names,
                                         title='Class Distribution in Test Set')
            plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 2. Confusion matrix (top classes only)
            fig = plot_confusion_matrix(
                targets, predictions,
                class_names=class_names,
                normalize=True,
                title='Normalized Confusion Matrix (Top Classes)',
                max_classes=10
            )
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 3. Prediction distribution
            fig = plot_class_distribution(pred_counter, class_names,
                                         title='Prediction Distribution')
            plt.savefig(os.path.join(self.output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            self._log(f"Saved visualization plots to {self.output_dir}")

        except Exception as e:
            self._log(f"Error generating plots: {str(e)}")

    def _save_confusion_matrix(self, targets, predictions):
        """Save confusion matrix visualization if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            # Get the most common classes in the targets
            target_counter = Counter(targets)
            top_classes = [idx for idx, _ in target_counter.most_common(10)]

            # Create class name mapping with consistent formatting
            class_names = {}
            if self.idx_to_chord:
                for cls in top_classes:
                    # Always use the chord mapping if available, never fall back to "Class-X" format
                    if cls in self.idx_to_chord:
                        class_names[cls] = self.idx_to_chord[cls]
                    else:
                        class_names[cls] = f"Unknown-{cls}"
            else:
                class_names = {cls: f"Class-{cls}" for cls in top_classes}

            # Also create a mapping for ALL classes in predictions for consistent reporting
            all_class_names = {}
            if self.idx_to_chord:
                # Build a complete mapping for all encountered classes
                for cls in set(list(targets) + list(predictions)):
                    if cls in self.idx_to_chord:
                        all_class_names[cls] = self.idx_to_chord[cls]
                    else:
                        all_class_names[cls] = f"Unknown-{cls}"
            else:
                all_class_names = {cls: f"Class-{cls}" for cls in set(list(targets) + list(predictions))}

            # Print detailed per-class accuracy analysis
            self._log("\nConfusion Matrix Analysis (Top Classes):")
            self._log(f"{'True Class':<20} | {'Accuracy':<10} | {'Most Predicted':<20} | {'Correct/Total'}")
            self._log(f"{'-'*20} | {'-'*10} | {'-'*20} | {'-'*15}")

            # Calculate per-class accuracy and most common predictions
            for cls in top_classes:
                # Get indices where the true class is this class
                mask = np.array(targets) == cls
                if not any(mask):
                    continue

                # Get the predictions for this class
                cls_preds = np.array(predictions)[mask]

                # Calculate accuracy for this class
                correct = (cls_preds == cls).sum()
                total = len(cls_preds)
                accuracy = correct / total if total > 0 else 0

                # Find most common prediction for this class
                pred_counter = Counter(cls_preds)
                most_common_pred, most_common_count = pred_counter.most_common(1)[0] if pred_counter else (None, 0)

                # Get the readable names using the consistent mapping
                cls_name = all_class_names.get(cls, f"Class-{cls}")
                most_common_name = all_class_names.get(most_common_pred, f"Class-{most_common_pred}") if most_common_pred is not None else "N/A"

                # Log the class stats
                self._log(f"{cls_name:<20} | {accuracy:.4f}     | {most_common_name:<20} | {correct}/{total}")

            # Filter data to include only top classes
            mask = np.isin(targets, top_classes)
            filtered_targets = np.array(targets)[mask]
            filtered_preds = np.array(predictions)[mask]

            # Compute confusion matrix
            cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)

            # Normalize the confusion matrix - FIX: Handle division by zero and NaN values
            row_sums = cm.sum(axis=1)
            # Add small epsilon to avoid division by zero
            row_sums = np.where(row_sums == 0, 1e-10, row_sums)
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]

            # Replace NaN values with zeros for better visualization
            cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

            # Create figure with robust min/max values
            plt.figure(figsize=(10, 8))

            # Use robust min/max values to avoid seaborn warnings
            vmin = np.nanmin(cm_normalized[~np.isnan(cm_normalized)]) if np.any(~np.isnan(cm_normalized)) else 0
            vmax = np.nanmax(cm_normalized[~np.isnan(cm_normalized)]) if np.any(~np.isnan(cm_normalized)) else 1

            # Get readable names for the confusion matrix labels using our consistent mapping
            x_labels = [all_class_names.get(cls, f"Class-{cls}") for cls in top_classes]
            y_labels = [all_class_names.get(cls, f"Class-{cls}") for cls in top_classes]

            # Create heatmap with robust parameters
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        vmin=vmin, vmax=vmax,
                        xticklabels=x_labels,
                        yticklabels=y_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Top 10 Classes)')
            plt.tight_layout()

            # Save if output directory is provided
            if self.output_dir:
                figures_dir = os.path.join(self.output_dir, "figures")
                os.makedirs(figures_dir, exist_ok=True)
                plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=300)
                self._log(f"Saved confusion matrix to {figures_dir}")

            plt.close()

        except Exception as e:
            self._log(f"Error creating confusion matrix: {e}")
