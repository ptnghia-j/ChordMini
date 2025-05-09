import torch
import torch.distributed as dist
import time
import os # Added for path join
from modules.training.StudentTrainer import StudentTrainer
from modules.utils.logger import info, warning, error, debug
from torch.nn.parallel import DistributedDataParallel
from modules.utils.timeout_handler import timeout_handler, TimeoutException
from modules.utils import checkpoint_utils # Added import

class DistributedStudentTrainer(StudentTrainer):
    """
    Extension of StudentTrainer that supports distributed training with PyTorch DDP.
    Handles metric aggregation across ranks and coordinates checkpoint saving.
    """

    def __init__(self, model, optimizer, device, num_epochs=100, logger=None,
                 checkpoint_dir='checkpoints', class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5, lr_decay_factor=0.95,
                 min_lr=5e-6, use_warmup=False, warmup_epochs=None, warmup_start_lr=None,
                 warmup_end_lr=None, lr_schedule_type=None, use_focal_loss=False,
                 focal_gamma=2.0, focal_alpha=None, use_kd_loss=False, kd_alpha=0.5,
                 temperature=1.0, rank=0, world_size=1, timeout_minutes=30):

        # guard invalid CUDA device ordinal
        if isinstance(device, torch.device) and device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            idx = device.index
            if idx is not None and idx >= gpu_count:
                warning(f"Invalid CUDA device index {idx}, falling back to CPU")
                device = torch.device('cpu')

        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            num_epochs=num_epochs,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            class_weights=class_weights,
            idx_to_chord=idx_to_chord,
            normalization=normalization,
            early_stopping_patience=early_stopping_patience,
            lr_decay_factor=lr_decay_factor,
            min_lr=min_lr,
            use_warmup=use_warmup,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            warmup_end_lr=warmup_end_lr,
            lr_schedule_type=lr_schedule_type,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            use_kd_loss=use_kd_loss,
            kd_alpha=kd_alpha,
            temperature=temperature
        )

        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.timeout_minutes = timeout_minutes

        # avoid DDP “mark variable ready twice” when using checkpointing
        if self.world_size > 1 and isinstance(self.model, DistributedDataParallel):
            info("Enabling static_graph on DDP to avoid reentrant-backward errors")
            self.model._set_static_graph()

        if self.is_main_process:
            info(f"Initialized DistributedStudentTrainer with {world_size} processes")
            info(f"This is the main process (rank {rank})")
            info(f"Using timeout of {timeout_minutes} minutes for distributed operations")

    def _get_model_state_dict(self):
        """Helper to get model state_dict, unwrapping DDP model if necessary."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def reduce_tensor(self, tensor, timeout_minutes=None):
        """Reduce tensor across all ranks with timeout handling.

        Args:
            tensor: Tensor to reduce
            timeout_minutes: Timeout in minutes (default: 30)

        Returns:
            Reduced tensor
        """
        rt = tensor.clone()

        # Use class timeout if not specified
        if timeout_minutes is None:
            timeout_minutes = self.timeout_minutes

        # Convert minutes to seconds
        timeout_seconds = timeout_minutes * 60

        try:
            # Use timeout handler for the all_reduce operation
            with timeout_handler(seconds=timeout_seconds,
                                error_message=f"Distributed all_reduce timed out after {timeout_minutes} minutes"):
                # Only log if there's an issue
                dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        except TimeoutException as e:
            warning(f"Rank {self.rank}: {str(e)}")
            warning(f"Rank {self.rank}: Using local tensor value instead")
            # Return the local tensor value if timeout occurs
            return tensor
        except Exception as e:
            error(f"Rank {self.rank}: Error in all_reduce: {str(e)}")
            # Return the local tensor value if any error occurs
            return tensor

        rt /= self.world_size
        return rt

    def train_epoch(self, train_loader, val_loader, epoch):
        start_time = time.time()
        """
        Train for one epoch with distributed support.
        Aggregates metrics across all ranks.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Set epoch for distributed samplers
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # normalize batch format: dict expected by train_batch
            if isinstance(batch, (tuple, list)):
                batch = {'spectro': batch[0], 'chord_idx': batch[1]}
            batch_result = self.train_batch(batch)

            # Update metrics
            total_loss += batch_result['loss']

            # Calculate accuracy from batch result
            if isinstance(batch, dict) and 'spectro' in batch:
                batch_size = batch['spectro'].size(0)
            else:
                # For tuple format (inputs, targets)
                batch_size = batch[0].size(0)
            correct += batch_result['accuracy'] * batch_size
            total += batch_size

            # Log progress
            if batch_idx % 100 == 0 and self.is_main_process:
                info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                     f'Loss: {batch_result["loss"]:.4f} | Acc: {batch_result["accuracy"]*100:.2f}%')

        # Aggregate metrics across all processes
        if self.world_size > 1:
            # Convert to tensors for reduction
            # Ensure tensors are float for reduction and averaging
            loss_tensor = torch.tensor(total_loss, dtype=torch.float, device=self.device)
            correct_tensor = torch.tensor(correct, dtype=torch.float, device=self.device)
            total_tensor = torch.tensor(total, dtype=torch.float, device=self.device)

            # Reduce tensors
            loss_tensor = self.reduce_tensor(loss_tensor)
            correct_tensor = self.reduce_tensor(correct_tensor)
            total_tensor = self.reduce_tensor(total_tensor)

            # Convert back to Python values
            total_loss = loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0

        # Log training metrics
        if self.is_main_process:
            info(f'Epoch: {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy*100:.2f}%')

        # report epoch elapsed time
        if self.is_main_process:
            elapsed = time.time() - start_time
            info(f'Epoch {epoch} time: {elapsed:.2f} sec')

        # Evaluate on validation set
        val_loss, val_acc = self.evaluate(val_loader, epoch)

        # Update learning rate based on validation performance
        if not self.lr_schedule_type:
            # Only adjust learning rate if not using a scheduler
            self._adjust_learning_rate(val_acc)
        else:
            # Update learning rate using scheduler
            self._update_learning_rate(epoch)

        # Check for early stopping
        early_stop = False
        if self._save_best_model(val_acc, val_loss, epoch): # This will be handled by overridden method
            if self.is_main_process: # Log only on main process
                info(f'New best model with validation accuracy: {val_acc:.4f}')

        if self._check_early_stopping(): # _check_early_stopping is fine as is
            if self.is_main_process:
                info('Early stopping triggered')
            early_stop = True

        return early_stop

    def evaluate(self, val_loader, epoch=None):
        """
        Evaluate the model on the validation set with distributed support.
        Aggregates metrics across all ranks.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # track per-sample preds & targets for class-wise accuracy
        all_preds = []
        all_targets = []

        # Set epoch for distributed samplers
        if hasattr(val_loader.sampler, 'set_epoch') and epoch is not None:
            val_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get input and target
                if isinstance(batch, dict) and 'spectro' in batch:
                    # Handle dictionary format from SynthDataset's iterator
                    spectro = batch['spectro'].to(self.device)
                    targets = batch['chord_idx'].to(self.device)

                    # Get teacher logits if using knowledge distillation
                    teacher_logits = None
                    if self.use_kd_loss and 'teacher_logits' in batch:
                        teacher_logits = batch['teacher_logits'].to(self.device)
                else:
                    # Handle tuple format from distributed DataLoader
                    spectro, targets = batch
                    spectro = spectro.to(self.device)
                    targets = targets.to(self.device)

                    # No teacher logits in this format
                    teacher_logits = None

                # Normalize input
                if self.normalization:
                    spectro = (spectro - self.normalization['mean']) / self.normalization['std']

                # Forward pass
                outputs = self.model(spectro)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # --- Flatten per-frame logits/targets AND teacher_logits consistently ---
                logits_for_loss = logits
                targets_for_loss = targets
                teacher_logits_for_loss = teacher_logits

                if logits.ndim == 3 and targets.ndim == 2:
                    batch_size, time_steps, num_classes = logits.shape
                    debug(f"Eval: Original shapes: logits {logits.shape}, targets {targets.shape}")

                    # Flatten student logits and targets for loss
                    logits_for_loss = logits.reshape(-1, num_classes)
                    targets_for_loss = targets.reshape(-1)
                    debug(f"Eval: Flattened student for loss: logits {logits_for_loss.shape}, targets {targets_for_loss.shape}")

                    # Flatten teacher logits if they exist and are 3D
                    if teacher_logits is not None and teacher_logits.ndim == 3:
                        if teacher_logits.shape[0] == batch_size and teacher_logits.shape[1] == time_steps:
                            teacher_logits_for_loss = teacher_logits.reshape(-1, teacher_logits.size(-1))
                            debug(f"Eval: Flattened teacher logits for loss: {teacher_logits_for_loss.shape}")
                        else:
                            warning(f"Eval: Teacher logits shape {teacher_logits.shape} mismatch with student batch/time {batch_size}/{time_steps}. Cannot flatten consistently.")
                            teacher_logits_for_loss = None # Invalidate teacher logits for loss calculation
                # ----------------------------------------------------------------------

                # Calculate loss using potentially flattened tensors
                # Use the compute_loss method from the parent class (StudentTrainer)
                # which now handles KD and flattening internally
                loss = self.compute_loss(logits_for_loss, targets_for_loss, teacher_logits_for_loss)

                total_loss += loss.item()

                # --- Calculate accuracy using potentially flattened tensors ---
                # Use the same flattened tensors as used for loss calculation
                _, predicted = logits_for_loss.max(1)
                batch_correct = (predicted == targets_for_loss).sum().item()
                batch_total = targets_for_loss.size(0)

                # collect for class‐wise metrics (using the same flattened predictions/targets)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_targets.extend(targets_for_loss.cpu().numpy().tolist())

                correct += batch_correct
                total += batch_total

        # Aggregate metrics across all processes
        if self.world_size > 1:
            # Convert to tensors for reduction
            # Ensure tensors are float for reduction and averaging
            loss_tensor = torch.tensor(total_loss, dtype=torch.float, device=self.device)
            correct_tensor = torch.tensor(correct, dtype=torch.float, device=self.device)
            total_tensor = torch.tensor(total, dtype=torch.float, device=self.device)

            # Reduce tensors
            loss_tensor = self.reduce_tensor(loss_tensor)
            correct_tensor = self.reduce_tensor(correct_tensor)
            total_tensor = self.reduce_tensor(total_tensor)

            # Convert back to Python values
            total_loss = loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Log validation metrics
        if self.is_main_process:
            info(f'Validation Loss: {avg_loss:.4f} | Validation Acc: {accuracy*100:.2f}%')

            # Per-quality accuracy calculation (ensure idx_to_chord exists)
            if self.idx_to_chord:
                info("Per-quality accuracy:")
                try:
                    # --- MODIFICATION START: Update quality extraction logic ---
                    idx_to_quality = {}
                    for idx, name in self.idx_to_chord.items():
                        if isinstance(name, str):
                            if name == "N" or name == "X":
                                continue # Skip special chords N and X for quality mapping
                            elif ':' in name:
                                quality = name.split(':')[-1]
                                idx_to_quality[idx] = quality
                            else:
                                # Assume it's a major chord if no colon and not N/X
                                idx_to_quality[idx] = "maj"
                    # --- MODIFICATION END ---

                    # Filter out invalid indices from predictions and targets before mapping
                    # This filtering might not be strictly necessary anymore with the improved mapping,
                    # but keeping it adds robustness against potential future mapping issues.

                    # Map valid predictions and targets to qualities - This part seems overly complex and potentially buggy.
                    # Let's simplify the counting logic below instead.
                    # pred_quals = [idx_to_quality[p] for p in valid_preds if p in idx_to_quality] # Map only valid quality indices
                    # targ_quals = [idx_to_quality[t] for t in valid_targets if t in idx_to_quality] # Map only valid quality indices

                    # Ensure pred_quals and targ_quals have the same length after filtering - This is problematic if filtering differs
                    # min_len = min(len(pred_quals), len(targ_quals))
                    # pred_quals = pred_quals[:min_len]
                    # targ_quals = targ_quals[:min_len]

                    # Get all unique quality names found in the mapping
                    all_qualities = sorted(list(set(idx_to_quality.values())))

                    # Create a mapping from index to quality for all targets and predictions
                    quality_counts = {}
                    correct_counts = {}
                    # --- MODIFICATION START: Remove tracking for skipped indices ---
                    # skipped_indices = {} # No longer needed with improved mapping
                    # --- MODIFICATION END ---

                    # Find indices for special chord types "N" and "X"
                    n_chord_idx = None
                    x_chord_idx = None
                    for idx, name in self.idx_to_chord.items():
                        if name == "N":
                            n_chord_idx = idx
                            debug(f"Found 'N' chord at index {idx}")
                        elif name == "X":
                            x_chord_idx = idx
                            debug(f"Found 'X' chord at index {idx}")

                    # Log the special chord indices for debugging
                    info(f"Special chord indices - N: {n_chord_idx}, X: {x_chord_idx}")

                    # Count occurrences and correct predictions for each quality
                    for pred, target in zip(all_preds, all_targets):
                        target_qual = None
                        pred_qual = None

                        # Determine target quality
                        if target == n_chord_idx:
                            target_qual = "N"
                        elif target == x_chord_idx:
                            target_qual = "X"
                        elif target in idx_to_quality:
                            target_qual = idx_to_quality[target]
                        else:
                            # This case should ideally not happen with the improved mapping
                            warning(f"Target index {target} not found in quality mapping or special chords. Skipping.")
                            continue # Skip this sample if target quality cannot be determined

                        # Count this target quality
                        quality_counts[target_qual] = quality_counts.get(target_qual, 0) + 1

                        # Determine predicted quality
                        if pred == n_chord_idx:
                            pred_qual = "N"
                        elif pred == x_chord_idx:
                            pred_qual = "X"
                        elif pred in idx_to_quality:
                            pred_qual = idx_to_quality[pred]
                        # else: pred_qual remains None if prediction is not a valid quality index

                        # Check if prediction quality matches target quality
                        if pred_qual == target_qual:
                            correct_counts[target_qual] = correct_counts.get(target_qual, 0) + 1


                    # Add special chord types to the list of qualities to report if they exist
                    all_qualities_with_special = list(all_qualities)
                    if "N" in quality_counts:
                        if "N" not in all_qualities_with_special: all_qualities_with_special.append("N")
                    if "X" in quality_counts:
                        if "X" not in all_qualities_with_special: all_qualities_with_special.append("X")
                    # Ensure consistent sorting
                    all_qualities_with_special.sort()


                    # Log the distribution of chord qualities in the validation set
                    total_samples = sum(quality_counts.values())
                    # --- MODIFICATION START: Update log message ---
                    info(f"Validation set contains {total_samples} total samples across {len(quality_counts)} counted qualities")
                    # Remove logging for skipped counts as it's no longer relevant
                    # total_skipped = sum(skipped_indices.values())
                    # if total_skipped > 0:
                    #     warning(f"Skipped {total_skipped} samples because their target indices were not found in the quality mapping.")
                    #     # Log top 5 skipped indices for debugging
                    #     top_skipped = sorted(skipped_indices.items(), key=lambda item: item[1], reverse=True)[:5]
                    #     warning(f"Top 5 skipped target indices: {top_skipped}")
                    # --- MODIFICATION END ---

                    # Log the top 5 most common qualities
                    top_qualities = sorted(quality_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    info("Top 5 most common chord qualities in validation set:")
                    for qual, count in top_qualities:
                        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                        info(f"  {qual}: {count} samples ({percentage:.2f}%)")

                    # Report accuracy for each quality
                    for qual in all_qualities_with_special:
                        # Get the total count for this quality
                        total_count = quality_counts.get(qual, 0)

                        if total_count > 0:
                            # Get correct count for this quality
                            correct_count = correct_counts.get(qual, 0)
                            info(f'  {qual}: {correct_count/total_count:.2%} ({correct_count}/{total_count})')
                        else:
                            info(f'  {qual}: N/A (no samples)')
                except Exception as e:
                    error(f"Error calculating per-quality accuracy: {e}")
            else:
                warning("idx_to_chord mapping not available, skipping per-quality accuracy.")


        return avg_loss, accuracy

    def _save_best_model(self, val_acc, val_loss, epoch):
        """
        Save the model when validation accuracy improves.
        Only the main process (rank 0) saves checkpoints.
        """
        if not self.is_main_process:
            # For non-main processes, still update internal state if accuracy improved,
            # but don't save. The actual saving is guarded by is_main_process.
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            return False # Indicate no save happened on this process

        # Main process proceeds with saving logic from parent,
        # which now uses _get_model_state_dict() correctly.
        return super()._save_best_model(val_acc, val_loss, epoch)

    def _save_trainer_state(self, current_epoch_completed):
        """
        Save the full trainer state. Only the main process saves.
        """
        if not self.is_main_process:
            return False
        return super()._save_trainer_state(current_epoch_completed)

    def _save_epoch_checkpoint(self, epoch, avg_train_loss, train_acc):
        """
        Saves a model checkpoint for a specific epoch. Only the main process saves.
        """
        if not self.is_main_process:
            return
        super()._save_epoch_checkpoint(epoch, avg_train_loss, train_acc)


    def train(self, train_loader, val_loader, start_epoch=1):
        """
        Train the model for multiple epochs with distributed support.
        Only the main process logs progress.
        """
        # Ensure all processes are synchronized before starting training
        if self.world_size > 1:
            dist.barrier()

        # Only the main process logs the start of training
        if self.is_main_process:
            info(f"Starting distributed training with {self.world_size} processes")
            info(f"Training for {self.num_epochs} epochs starting from epoch {start_epoch}")

        # Train for the specified number of epochs
        for epoch in range(start_epoch, self.num_epochs + 1):
            # Train for one epoch
            early_stop = self.train_epoch(train_loader, val_loader, epoch)

            # Check for early stopping
            if early_stop:
                if self.is_main_process:
                    info(f"Early stopping triggered at epoch {epoch}")
                break

            # Ensure all processes are synchronized at the end of each epoch
            if self.world_size > 1:
                dist.barrier()

        # Only the main process logs the end of training
        if self.is_main_process:
            info("Training completed!")
