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
    
    def evaluate(self, save_plots=True):
        """
        Evaluate the model on the test set and generate performance metrics.
        
        Args:
            save_plots: Whether to save visualization plots
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        # Debug counters
        pred_counter = Counter()
        target_counter = Counter()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Extract inputs and targets
                if isinstance(batch, dict):
                    inputs = batch['spectro'].to(self.device)
                    targets = batch['chord_idx'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                # Apply normalization if provided
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']

                # Debug first few batches
                if batch_idx == 0:
                    self._log(f"Input shape: {inputs.shape}, target shape: {targets.shape}")
                    self._log(f"First few targets: {targets[:10].cpu().numpy()}")

                # Get raw logits before prediction
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Check if logits have reasonable values
                if batch_idx == 0:
                    self._log(f"Logits shape: {logits.shape}")
                    self._log(f"Logits mean: {logits.mean().item()}, std: {logits.std().item()}")
                    self._log(f"First batch sample logits (max 5 values): {logits[0, :5].cpu().numpy()}")

                # Use per-frame predictions if targets have a time dimension
                use_per_frame = targets.dim() > 1 and targets.shape[1] > 1
                
                if hasattr(self.model, 'predict'):
                    preds = self.model.predict(inputs, per_frame=use_per_frame)
                else:
                    # Fallback if no predict method
                    if logits.dim() == 3:  # [batch, time, classes]
                        preds = torch.argmax(logits, dim=2)
                    else:
                        preds = torch.argmax(logits, dim=1)

                # Process predictions based on their dimensions
                if targets.dim() > 1:
                    if preds.dim() > 1:
                        # Both are frame-level: flatten both
                        preds_np = preds.cpu().numpy().flatten()
                        targets_np = targets.cpu().numpy().flatten()
                    else:
                        # Targets are frame-level but preds are segment-level
                        # Use most common target for each sequence
                        seq_targets = []
                        for i in range(targets.shape[0]):
                            labels, counts = torch.unique(targets[i], return_counts=True)
                            most_common_idx = torch.argmax(counts)
                            seq_targets.append(labels[most_common_idx].item())
                        targets_np = np.array(seq_targets)
                        preds_np = preds.cpu().numpy()
                else:
                    # Standard case - both are segment-level
                    preds_np = preds.cpu().numpy().flatten()
                    targets_np = targets.cpu().numpy().flatten()

                # Count distribution using the adjusted arrays
                pred_counter.update(preds_np.tolist())
                target_counter.update(targets_np.tolist())

                all_preds.extend(preds_np)
                all_targets.extend(targets_np)

                # Debug first batch predictions vs targets
                if batch_idx == 0:
                    self._log(f"First batch - Predictions: {preds_np[:10]}")
                    self._log(f"First batch - Targets: {targets_np[:10]}")

        # Convert to numpy arrays for metrics calculation
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Print distribution statistics
        self._log("\nTarget Distribution (top 10):")
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            self._log(f"Target {idx} ({chord_name}): {count} occurrences ({count/len(all_targets)*100:.2f}%)")

        self._log("\nPrediction Distribution (top 10):")
        for idx, count in pred_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            self._log(f"Prediction {idx} ({chord_name}): {count} occurrences ({count/len(all_preds)*100:.2f}%)")

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Calculate per-class metrics for common classes
        self._analyze_confusion_matrix(all_targets, all_preds, target_counter)

        # Generate and save plots if requested
        if save_plots:
            self._generate_plots(all_targets, all_preds, target_counter, pred_counter)

        # Print overall metrics
        self._log(f"\nTest Metrics:")
        self._log(f"Test Accuracy: {accuracy:.4f}")
        self._log(f"Test Precision: {precision:.4f}")
        self._log(f"Test Recall: {recall:.4f}")
        self._log(f"Test F1 Score: {f1:.4f}")
        
        # Return metrics dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'target_distribution': target_counter,
            'prediction_distribution': pred_counter
        }
    
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
