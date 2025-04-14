import json
import traceback
import multiprocessing
import sys
import os
import torch  
import numpy as np
import argparse
import glob
import gc
from pathlib import Path

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, clear_gpu_cache
from modules.data.CrossValidationDataset import CrossValidationDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester
from modules.utils.teacher_utils import load_btc_model, extract_logits_from_teacher, generate_teacher_predictions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a chord recognition model with cross-validation and knowledge distillation")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/cv_kd', 
                        help='Directory to save checkpoints')
    parser.add_argument('--kfold', type=int, default=0, 
                        help='Which fold to use for validation (0-4)')
    parser.add_argument('--total_folds', type=int, default=5, 
                        help='Total number of folds')
    parser.add_argument('--storage_root', type=str, default=None, 
                        help='Root directory for data storage')
    parser.add_argument('--use_voca', action='store_true',
                       help='Use large vocabulary (170 chord types)')
    parser.add_argument('--teacher_model', type=str, default=None, 
                        help='Path to teacher model for knowledge distillation')
    parser.add_argument('--use_kd_loss', action='store_true',
                        help='Use knowledge distillation loss')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softening distributions (default: 1.0)')
    parser.add_argument('--audio_dirs', type=str, nargs='+', default=None,
                      help='Directories containing audio files')
    parser.add_argument('--label_dirs', type=str, nargs='+', default=None,
                      help='Directories containing label files')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache extracted features')
    parser.add_argument('--learning_rate', type=float, default=None, 
                        help='Base learning rate (overrides config value)')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size for training (overrides config value)')
    parser.add_argument('--preprocess', action='store_true',
                      help='Run preprocessing step to generate all features')
    parser.add_argument('--log_chord_details', action='store_true',
                       help='Enable detailed logging of chords during MIR evaluation')
    
    args = parser.parse_args()

    # Load configuration
    config = HParams.load(args.config)

    # Override with command line args
    if args.log_chord_details:
        if 'misc' not in config: config['misc'] = {}
        config.misc['log_chord_details'] = True
        logger.info("Detailed chord logging during evaluation ENABLED via command line.")
    elif config.misc.get('log_chord_details'):
        logger.info("Detailed chord logging during evaluation ENABLED via config/env.")

    # Set up device
    if config.misc.get('use_cuda', True) and is_cuda_available():
        device = get_device()
        logger.info(f"CUDA available. Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available or not requested. Using CPU.")
    
    # Override config values with command line arguments - IMPROVED CONFIG HANDLING
    config.misc['seed'] = args.seed if args.seed is not None else config.misc.get('seed', 42)
    config.paths['checkpoints_dir'] = args.save_dir if args.save_dir else config.paths.get('checkpoints_dir', './checkpoints/cv_kd')
    config.paths['storage_root'] = args.storage_root if args.storage_root else config.paths.get('storage_root', None)
    
    if args.learning_rate is not None:
        config.training['learning_rate'] = float(args.learning_rate)
    
    if args.batch_size is not None:
        config.training['batch_size'] = int(args.batch_size)
    
    # IMPROVED: Handle knowledge distillation settings properly with type conversion
    use_kd_loss = args.use_kd_loss
    if not use_kd_loss:
        # Check if config has this value as a string that needs conversion
        config_kd = config.training.get('use_kd_loss', False)
        if isinstance(config_kd, str):
            use_kd_loss = config_kd.lower() == "true"
        else:
            use_kd_loss = bool(config_kd)
    
    # Log KD settings explicitly
    if use_kd_loss:
        logger.info("Knowledge Distillation Loss ENABLED")
        kd_alpha = float(args.kd_alpha) if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
        temperature = float(args.temperature) if args.temperature is not None else float(config.training.get('temperature', 1.0))
        logger.info(f"KD Alpha: {kd_alpha}, Temperature: {temperature}")
    else:
        logger.info("Knowledge Distillation Loss DISABLED")
    
    # Set up chord mapping
    # Determine if large vocabulary is used
    use_large_voca = args.use_voca or config.feature.get('large_voca', False)

    if use_large_voca:
        logger.info("Using large vocabulary chord mapping (170 chords)")
        master_mapping = idx2voca_chord() # Get idx -> chord mapping
        chord_mapping = {chord: idx for idx, chord in master_mapping.items()} # Create reverse mapping
        n_classes = 170
    else:
        # Use the 25-chord vocabulary mapping (major/minor + no chord)
        logger.info("Using standard vocabulary chord mapping (25 chords)")
        chord_mapping = {} # chord -> idx
        master_mapping = {} # idx -> chord
        for i in range(12):
            root = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
            # Major chords
            maj_idx = i * 2
            maj_chord = f"{root}:maj" # Use explicit :maj
            chord_mapping[maj_chord] = maj_idx
            master_mapping[maj_idx] = maj_chord
            # Minor chords
            min_idx = i * 2 + 1
            min_chord = f"{root}:min"
            chord_mapping[min_chord] = min_idx
            master_mapping[min_idx] = min_chord
        # Special chords: N (no chord) and X (unknown)
        chord_mapping["N"] = 24
        master_mapping[24] = "N"
        chord_mapping["X"] = 25
        master_mapping[25] = "X"
        n_classes = 26 # 0-25 inclusive

    # Initialize chord class with the reverse mapping (chord -> idx)
    chord_class = Chords()
    chord_class.set_chord_mapping(chord_mapping)
    chord_class.initialize_chord_mapping() # Initialize variants

    # Log mapping info
    logger.info(f"\nUsing idx->chord mapping with {len(master_mapping)} entries")
    logger.info(f"Sample idx->chord mapping: {dict(list(master_mapping.items())[:5])}")
    logger.info(f"Reverse chord->idx mapping created with {len(chord_mapping)} entries")
    logger.info(f"Sample chord->idx mapping: {dict(list(chord_mapping.items())[:5])}")
    
    # Set random seed for reproducibility - ensure this is an integer
    seed = int(config.misc['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Create save directory
    save_dir = config.paths['checkpoints_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up datasets
    if args.audio_dirs:
        audio_dirs = args.audio_dirs
    else:
        # Default to the directories used in the teacher model
        audio_dirs = [
            os.path.join(config.paths.get('root_path', '/data/music'), 'isophonic'),
            os.path.join(config.paths.get('root_path', '/data/music'), 'uspop'),
            os.path.join(config.paths.get('root_path', '/data/music'), 'robbiewilliams')
        ]
    
    if args.label_dirs:
        label_dirs = args.label_dirs
    else:
        # Default to the same directories as audio (teacher model keeps labels with audio)
        label_dirs = audio_dirs
    
    # Cache directory for extracted features
    cache_dir = args.cache_dir or os.path.join(config.paths.get('root_path', '/data/music'), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load teacher model if provided
    teacher_model = None
    teacher_mean = None
    teacher_std = None
    
    if args.use_kd_loss and args.teacher_model:
        logger.info(f"Loading teacher model from {args.teacher_model}")
        try:
            # Determine vocabulary size based on args and config
            use_voca = args.use_voca or config.feature.get('large_voca', False)
            
            # Check if the teacher model path exists
            if not os.path.exists(args.teacher_model):
                # Try resolving with storage_root
                if config.paths.get('storage_root'):
                    alt_path = os.path.join(config.paths.get('storage_root'), args.teacher_model)
                    if os.path.exists(alt_path):
                        args.teacher_model = alt_path
                        logger.info(f"Resolved teacher model path to: {args.teacher_model}")
            
            # Load the teacher model using our utility function
            teacher_model, teacher_mean, teacher_std = load_btc_model(
                args.teacher_model, 
                device, 
                use_voca=use_voca
            )
            logger.info("Teacher model loaded successfully for knowledge distillation")
        except Exception as e:
            logger.error(f"Error loading teacher model: {e}")
            logger.error(traceback.format_exc())
            teacher_model = None
    
    # Run preprocessing if specified
    if args.preprocess:
        logger.info("Running preprocessing step to generate all features")
        # Create a dataset that will do all the preprocessing
        preprocess_dataset = CrossValidationDataset(
            config=config,
            audio_dirs=audio_dirs,
            label_dirs=label_dirs,
            chord_mapping=chord_mapping,
            train=True,  # Doesn't matter for preprocessing
            kfold=args.kfold,
            total_folds=args.total_folds,
            cache_dir=cache_dir,
            random_seed=seed,
            device=device
        )
        
        # Analyze label files before processing
        logger.info("Analyzing label files to understand structure...")
        preprocess_dataset.analyze_label_files(num_files=10)
        
        # Generate all features
        preprocess_dataset.generate_all_features()
        logger.info("Preprocessing completed")
        return
    
    # Create datasets for training and validation
    logger.info(f"Creating datasets for fold {args.kfold} of {args.total_folds}")
    
    # Initialize datasets without teacher model first
    train_dataset = CrossValidationDataset(
        config=config,
        audio_dirs=audio_dirs,
        label_dirs=label_dirs,
        chord_mapping=chord_mapping,
        train=True,
        kfold=args.kfold,
        total_folds=args.total_folds,
        cache_dir=cache_dir,
        random_seed=seed,
        device=device,
        teacher_model=None  # Don't pass teacher model yet
    )
    
    val_dataset = CrossValidationDataset(
        config=config,
        audio_dirs=audio_dirs,
        label_dirs=label_dirs,
        chord_mapping=chord_mapping,
        train=False,
        kfold=args.kfold,
        total_folds=args.total_folds,
        cache_dir=cache_dir,
        random_seed=seed,
        device=device
    )
    
    # After creating datasets
    logger.info("Analyzing chord distributions:")
    val_dataset.analyze_chord_distribution()
    train_dataset.analyze_chord_distribution()
    
    # Create data loaders
    batch_size = config.training.get('batch_size', 16)
    train_loader = train_dataset.get_data_loader(batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = val_dataset.get_data_loader(batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Generate teacher predictions for training data if using KD
    teacher_predictions = None
    if args.use_kd_loss and teacher_model is not None:
        logger.info("Generating teacher predictions for knowledge distillation")
        
        # Set up a directory to save teacher logits
        logits_dir = os.path.join(save_dir, f"teacher_logits_fold{args.kfold}")
        os.makedirs(logits_dir, exist_ok=True)
        
        # Generate predictions
        teacher_predictions = generate_teacher_predictions(
            teacher_model, 
            train_loader, 
            teacher_mean, 
            teacher_std,
            device,
            save_dir=logits_dir
        )
        
        logger.info(f"Generated teacher predictions for {len(teacher_predictions)} samples")
    
    # Calculate global mean and std
    logger.info("Calculating global mean and std")
    mean = 0.0
    square_mean = 0.0
    k = 0
    
    mean_std_cache = os.path.join(cache_dir, f'normalization_fold{args.kfold}.pt')
    if os.path.exists(mean_std_cache):
        # Load from cache
        normalization_data = torch.load(mean_std_cache)
        mean = normalization_data.get('mean', 0.0)
        std = normalization_data.get('std', 1.0)
        logger.info(f"Loaded normalization from cache: mean={mean}, std={std}")
    else:
        # Calculate from data
        temp_loader = train_dataset.get_data_loader(batch_size=batch_size, shuffle=True, num_workers=2)
        for i, data in enumerate(temp_loader):
            features = data['spectro'].to('cpu')
            mean += torch.mean(features).item()
            square_mean += torch.mean(features.pow(2)).item()
            k += 1
            if i >= 99:
                break

        if k > 0:
            square_mean = square_mean / k
            mean = mean / k
            std = np.sqrt(max(0, square_mean - mean * mean))
            if std == 0: std = 1.0

            normalization_data = {'mean': mean, 'std': std}
            torch.save(normalization_data, mean_std_cache)
            logger.info(f"Calculated normalization: mean={mean:.4f}, std={std:.4f}")
        else:
            logger.warning("Could not calculate normalization stats (k=0). Using defaults.")
            mean = 0.0
            std = 1.0
    
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32)
    normalization = {'mean': mean_tensor, 'std': std_tensor}
    logger.info(f"Normalization tensors created on device: {device}")
    
    # Create model
    logger.info(f"Creating model with {n_classes} output classes")
    
    # Log additional information about feature dimensions
    n_freq = config.feature.get('n_bins', 144)
    n_group = config.model.get('n_group', 32)
    feature_dim = n_freq // n_group if n_group > 0 else n_freq
    heads = config.model.get('f_head', 6)
    logger.info(f"Using feature dimensions: n_freq={n_freq}, n_group={n_group}, feature_dim={feature_dim}, heads={heads}")
    
    model = ChordNet(
        n_freq=config.feature.get('n_bins', 144),
        n_classes=n_classes,
        n_group=config.model.get('n_group', 32),
        f_layer=config.model.get('f_layer', 3),
        f_head=config.model.get('f_head', 6),
        t_layer=config.model.get('t_layer', 3),
        t_head=config.model.get('t_head', 6),
        d_layer=config.model.get('d_layer', 3),
        d_head=config.model.get('d_head', 6),
        dropout=config.model.get('dropout', 0.3)
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.get('learning_rate', 0.0001),
        weight_decay=config.training.get('weight_decay', 0.0)
    )
    
    # IMPROVED: Handle string representations of boolean values
    use_focal_loss = args.use_focal_loss
    if not use_focal_loss:
        config_focal = config.training.get('use_focal_loss', False)
        if isinstance(config_focal, str):
            use_focal_loss = config_focal.lower() == "true"
        else:
            use_focal_loss = bool(config_focal)
            
    focal_gamma = float(args.focal_gamma) if args.focal_gamma is not None else float(config.training.get('focal_gamma', 2.0))
    focal_alpha = float(args.focal_alpha) if args.focal_alpha is not None else config.training.get('focal_alpha')
    
    if use_focal_loss:
        logger.info(f"Using Focal Loss with gamma={focal_gamma}")
        if focal_alpha is not None:
            logger.info(f"Focal Loss alpha={focal_alpha}")
    
    # Create trainer with knowledge distillation - with properly typed parameters
    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config.training.get('num_epochs', 50)),
        logger=logger,
        checkpoint_dir=save_dir,
        class_weights=None,  # No class weights for now
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=int(config.training.get('early_stopping_patience', 5)),
        lr_decay_factor=float(config.training.get('lr_decay_factor', 0.95)),
        min_lr=float(config.training.get('min_learning_rate', 5e-6)),
        use_warmup=args.use_warmup if args.use_warmup else bool(config.training.get('use_warmup', False)),
        warmup_epochs=int(config.training.get('warmup_epochs', 5)) if config.training.get('warmup_epochs') else None,
        warmup_start_lr=float(config.training.get('warmup_start_lr', 1e-6)) if config.training.get('warmup_start_lr') else None,
        warmup_end_lr=float(config.training.get('warmup_end_lr', config.training.get('learning_rate', 0.0001))) if config.training.get('warmup_end_lr') else None,
        lr_schedule_type=config.training.get('lr_schedule_type'),
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_kd_loss=use_kd_loss,
        kd_alpha=kd_alpha,
        temperature=temperature,
        teacher_model=teacher_model,
        teacher_normalization={'mean': teacher_mean, 'std': teacher_std},
        teacher_predictions=teacher_predictions
    )
    
    # Set chord mapping
    trainer.set_chord_mapping(chord_mapping)
    
    # Train the model
    logger.info(f"Starting training for fold {args.kfold}")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    # Final evaluation on validation set (which is the test set for this fold)
    logger.info("\n=== Testing (Validation Fold) ===")
    try:
        if trainer.load_best_model():
            # Use the validation loader (acts as test loader for this fold)
            test_loader = val_loader # Already created earlier
            
            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader, # Use the validation loader here
                device=device,
                idx_to_chord=master_mapping, # Use master_mapping (idx->chord)
                normalization=normalization, # Use the calculated normalization
                output_dir=save_dir, # Use save_dir for this fold
                logger=logger
            )
            
            test_metrics = tester.evaluate(save_plots=True)
            
            # Save test metrics for this fold
            try:
                metrics_path = os.path.join(save_dir, f"test_metrics_fold{args.kfold}.json") 
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics for fold {args.kfold} saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics for fold {args.kfold}: {e}")

            # NEW: Generate chord quality distribution and accuracy visualization
            logger.info("\n=== Generating Chord Quality Distribution and Accuracy Graph (Validation Fold) ===")
            try:
                from modules.utils.visualize import plot_chord_quality_distribution_accuracy
                
                # Collect all predictions and targets from the validation set
                all_preds = []
                all_targets = []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader: # Iterate through validation loader
                        inputs, targets = batch['spectro'], batch['chord_idx']
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Apply normalization if available
                        if normalization and 'mean' in normalization and 'std' in normalization:
                            inputs = (inputs - normalization['mean']) / normalization['std']
                            
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                            
                        # Handle 3D logits (sequence data) - Average over time dimension for frame-level eval
                        if logits.ndim == 3 and targets.ndim == 2:
                             # Flatten logits and targets for frame-level comparison
                            logits = logits.view(-1, logits.size(-1)) # (batch*seq_len, n_classes)
                            targets = targets.view(-1) # (batch*seq_len)
                        elif logits.ndim == 3 and targets.ndim == 1:
                             # If targets are already flat, just flatten logits
                            logits = logits.view(-1, logits.size(-1))

                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                
                # Define focus qualities
                focus_qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7", 
                                  "min-maj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
                
                # Create distribution and accuracy visualization for this fold
                quality_dist_path = os.path.join(save_dir, f"chord_quality_distribution_accuracy_fold{args.kfold}.png")
                plot_chord_quality_distribution_accuracy(
                    all_preds, all_targets, master_mapping, # Use master_mapping (idx->chord)
                    save_path=quality_dist_path,
                    title=f"Chord Quality Distribution and Accuracy (Fold {args.kfold})",
                    focus_qualities=focus_qualities
                )
                logger.info(f"Chord quality distribution graph for fold {args.kfold} saved to {quality_dist_path}")
            except ImportError:
                 logger.warning("Could not import plot_chord_quality_distribution_accuracy. Skipping visualization.")
                 logger.warning("Install matplotlib and seaborn: pip install matplotlib seaborn")
            except Exception as e:
                logger.error(f"Error creating chord quality distribution graph for fold {args.kfold}: {e}")
                logger.error(traceback.format_exc())

            # Advanced testing with mir_eval module on validation set
            logger.info("\n=== MIR evaluation (Validation Fold) ===")
            score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
            
            # Use the validation samples for this fold
            validation_samples = val_dataset.samples # These are the samples for the current validation fold
            dataset_length = len(validation_samples)
            
            if dataset_length < 3:
                logger.info(f"Not enough validation samples ({dataset_length}) in fold {args.kfold} to compute chord metrics with 3 splits.")
                 # Optionally run on the whole validation set if splits aren't possible
                if dataset_length > 0:
                     logger.info(f"Evaluating model on all {dataset_length} validation samples for fold {args.kfold}...")
                     score_list_dict, song_length_list, average_score_dict = large_voca_score_calculation(
                         valid_dataset=validation_samples, config=config, model=model, model_type='ChordNet',
                         mean=mean, std=std, device=device)
                     mir_eval_results = average_score_dict # Store the single result
                     logger.info(f"MIR evaluation results for fold {args.kfold} (all samples): {mir_eval_results}")
                else:
                     logger.info(f"No validation samples available for MIR evaluation in fold {args.kfold}.")
                     mir_eval_results = {}
            else:
                # Create balanced splits from the validation samples
                split = dataset_length // 3
                valid_dataset1 = validation_samples[:split]
                valid_dataset2 = validation_samples[split:2*split]
                valid_dataset3 = validation_samples[2*split:]
                
                # Evaluate each split
                logger.info(f"Evaluating model on {len(valid_dataset1)} validation samples in split 1 (Fold {args.kfold})...")
                score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(
                    valid_dataset=valid_dataset1, config=config, model=model, model_type='ChordNet', 
                    mean=mean, std=std, device=device)
                
                logger.info(f"Evaluating model on {len(valid_dataset2)} validation samples in split 2 (Fold {args.kfold})...")
                score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(
                    valid_dataset=valid_dataset2, config=config, model=model, model_type='ChordNet', 
                    mean=mean, std=std, device=device)
                
                logger.info(f"Evaluating model on {len(valid_dataset3)} validation samples in split 3 (Fold {args.kfold})...")
                score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(
                    valid_dataset=valid_dataset3, config=config, model=model, model_type='ChordNet', 
                    mean=mean, std=std, device=device)
                
                # Calculate weighted averages
                mir_eval_results = {}
                # Check if all results are valid before calculating average
                valid_results = (average_score_dict1 and average_score_dict2 and average_score_dict3 and
                                 song_length_list1 and song_length_list2 and song_length_list3)

                if valid_results:
                    total_length = np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3)
                    if total_length > 0:
                        for m in score_metrics:
                             # Ensure metric exists in all dictionaries
                            if m in average_score_dict1 and m in average_score_dict2 and m in average_score_dict3:
                                # Calculate weighted average based on song lengths
                                avg = (np.sum(song_length_list1) * average_score_dict1[m] +
                                       np.sum(song_length_list2) * average_score_dict2[m] +
                                       np.sum(song_length_list3) * average_score_dict3[m]) / total_length
                                
                                # Log individual split scores
                                logger.info(f"==== (Fold {args.kfold}) {m} score 1: {average_score_dict1[m]:.4f}")
                                logger.info(f"==== (Fold {args.kfold}) {m} score 2: {average_score_dict2[m]:.4f}")
                                logger.info(f"==== (Fold {args.kfold}) {m} score 3: {average_score_dict3[m]:.4f}")
                                logger.info(f"==== (Fold {args.kfold}) {m} weighted average: {avg:.4f}")
                                
                                # Store in results dictionary
                                mir_eval_results[m] = {
                                    'split1': float(average_score_dict1[m]),
                                    'split2': float(average_score_dict2[m]),
                                    'split3': float(average_score_dict3[m]),
                                    'weighted_avg': float(avg)
                                }
                            else:
                                logger.warning(f"(Fold {args.kfold}) Metric '{m}' missing in one or more splits, cannot calculate average.")
                                mir_eval_results[m] = {'error': f'Metric {m} missing in splits'}
                    else:
                        logger.warning(f"(Fold {args.kfold}) Total song length is zero, cannot calculate weighted average MIR scores.")
                        mir_eval_results = {'error': 'Total song length zero'}
                else:
                    logger.warning(f"(Fold {args.kfold}) Could not calculate MIR scores properly for all splits.")
                    mir_eval_results = {'error': 'Calculation failed for one or more splits'}

            # Save MIR-eval metrics for this fold
            try:
                mir_eval_path = os.path.join(save_dir, f"mir_eval_metrics_fold{args.kfold}.json")
                with open(mir_eval_path, 'w') as f:
                    json.dump(mir_eval_results, f, indent=2)
                logger.info(f"MIR evaluation metrics for fold {args.kfold} saved to {mir_eval_path}")
            except Exception as e:
                 logger.error(f"Error saving MIR evaluation metrics for fold {args.kfold}: {e}")
                
        else:
            logger.warning(f"Could not load best model for testing fold {args.kfold}")
    except Exception as e:
        logger.error(f"Error during testing for fold {args.kfold}: {e}")
        logger.error(traceback.format_exc())
    
    # Save final model for the fold (keep this part)
    try:
        final_path = os.path.join(save_dir, f'final_model_fold{args.kfold}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'fold': args.kfold,
            'chord_mapping': chord_mapping,
            'mean': mean,
            'std': std
        }, final_path)
        logger.info(f"Final model saved to {final_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set or not available
        pass
    main()
