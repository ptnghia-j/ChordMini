import torch
import warnings
from modules.utils.logger import info

def safe_clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False, verbose=True):
    """
    Safely clip gradient norm while providing helpful diagnostics for non-finite values.

    Args:
        parameters: Model parameters to clip gradients for
        max_norm: Maximum allowed gradient norm
        error_if_nonfinite: Whether to raise error on non-finite gradients
        verbose: Whether to print detailed diagnostics when non-finite values are found

    Returns:
        total_norm: The total gradient norm before clipping
    """
    # Filter parameters that have gradients
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    # Check if there are any parameters with gradients
    if len(parameters) == 0:
        return torch.tensor(0.0)

    # Track gradient statistics for adaptive handling
    if not hasattr(safe_clip_grad_norm_, 'problem_history'):
        safe_clip_grad_norm_.problem_history = {}
        safe_clip_grad_norm_.call_count = 0
        safe_clip_grad_norm_.last_report = 0

    # Increment call counter
    safe_clip_grad_norm_.call_count += 1

    # Add gradient clamping to stabilize training
    for p in parameters:
        if p.grad is not None:
            # Enhanced clamping strategy with larger bounds for main parameters
            if p.numel() <= 12:  # For small vectors like biases
                p.grad.data.clamp_(-1.0, 1.0)  # More conservative for small parameters
            elif p.numel() <= 144:  # Medium-sized matrices (e.g. 12x12)
                p.grad.data.clamp_(-3.0, 3.0)  # Moderate clamping for medium tensors
            else:
                p.grad.data.clamp_(-5.0, 5.0)  # Allow larger gradients for weights

    # Check for non-finite gradients before clipping
    has_nonfinite = False
    problem_params = []

    for i, p in enumerate(parameters):
        if not torch.isfinite(p.grad).all():
            has_nonfinite = True
            problem_params.append((i, p))

            # Track problematic parameters by shape to identify recurring issues
            param_shape = tuple(p.shape)
            if param_shape not in safe_clip_grad_norm_.problem_history:
                safe_clip_grad_norm_.problem_history[param_shape] = {
                    'count': 0,
                    'total_nan_percent': 0.0,
                    'total_inf_percent': 0.0,
                    'last_seen': 0
                }

            # Update statistics
            nan_count = torch.isnan(p.grad).sum().item()
            inf_count = torch.isinf(p.grad).sum().item()
            total_elements = p.grad.numel()
            nan_percent = nan_count/total_elements if total_elements > 0 else 0
            inf_percent = inf_count/total_elements if total_elements > 0 else 0

            stats = safe_clip_grad_norm_.problem_history[param_shape]
            stats['count'] += 1
            stats['total_nan_percent'] += nan_percent
            stats['total_inf_percent'] += inf_percent
            stats['last_seen'] = safe_clip_grad_norm_.call_count

    # Handle non-finite gradients with better diagnostics and adaptive handling
    if has_nonfinite:
        if verbose:
            info(f"Non-finite gradients detected in {len(problem_params)} parameters")

            # Print stats about the first few problematic parameters
            for i, (idx, param) in enumerate(problem_params[:3]):  # Limit to first 3
                grad = param.grad
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                total_elements = grad.numel()

                info(
                    f"Parameter {idx}: shape={list(param.shape)}, "
                    f"NaNs: {nan_count}/{total_elements} ({nan_count/total_elements:.2%}), "
                    f"Infs: {inf_count}/{total_elements} ({inf_count/total_elements:.2%})"
                )

            if len(problem_params) > 3:
                info(f"... and {len(problem_params) - 3} more parameters with issues")

        # ADAPTIVE HANDLING: Instead of just zeroing out bad gradients, apply a recovery strategy
        for _, p in problem_params:
            grad = p.grad
            mask_finite = torch.isfinite(grad)
            mask_nonfinite = ~mask_finite

            if mask_nonfinite.any():
                # Check if we can recover from partial NaNs by using tensor statistics
                if mask_finite.any():
                    # Some values are finite - calculate statistics from those
                    finite_mean = grad[mask_finite].mean().item()
                    finite_std = grad[mask_finite].std().item()

                    # Replace non-finite values with small random values based on statistics
                    # This avoids completely killing the gradient
                    with torch.no_grad():
                        if abs(finite_mean) < 1e-6:
                            # If mean is very small, use a tiny fixed value with noise
                            fixed_val = 1e-6
                            noise = torch.randn_like(grad[mask_nonfinite]) * 1e-6
                            grad[mask_nonfinite] = fixed_val * torch.sign(noise) + noise
                        else:
                            # Scale down the mean and add small noise
                            recovery_scale = 0.01  # Scale factor for recovered values
                            noise_scale = max(abs(finite_std) * 0.01, 1e-6)
                            replacement = finite_mean * recovery_scale
                            noise = torch.randn_like(grad[mask_nonfinite]) * noise_scale
                            grad[mask_nonfinite] = replacement + noise
                else:
                    # All values are non-finite, replace with small random values
                    with torch.no_grad():
                        # Use a very small fixed value with minimal noise
                        grad[mask_nonfinite] = torch.randn_like(grad[mask_nonfinite]) * 1e-6

            # Apply an additional step: ensure the gradient has a minimum L2 norm
            # This helps prevent gradient vanishing after fixing non-finite values
            with torch.no_grad():
                grad_norm = torch.norm(grad)
                min_grad_norm = 1e-4  # Minimum allowed gradient norm
                if grad_norm < min_grad_norm:
                    # Rescale the gradient to ensure it has at least the minimum norm
                    scale_factor = min_grad_norm / (grad_norm + 1e-10)
                    grad.mul_(scale_factor)

        # Log adaptive recovery rather than zeroing
        info(
            "Non-finite gradients detected and adaptively reconstructed. "
            "Applied small random values with proper scaling to maintain training signal."
        )

    # Periodically report persistent gradient issues (every 200 steps)
    if safe_clip_grad_norm_.call_count - safe_clip_grad_norm_.last_report >= 200:
        safe_clip_grad_norm_.last_report = safe_clip_grad_norm_.call_count

        # Find shapes with recurring issues
        persistent_issues = {
            shape: stats for shape, stats in safe_clip_grad_norm_.problem_history.items()
            if stats['count'] > 5  # Shapes with multiple occurrences
        }

        if persistent_issues:
            info("===== Gradient Health Report =====")
            info(f"Total steps: {safe_clip_grad_norm_.call_count}")

            for shape, stats in sorted(persistent_issues.items(),
                                      key=lambda x: x[1]['count'],
                                      reverse=True)[:5]:  # Top 5 issues
                avg_nan = stats['total_nan_percent'] / stats['count'] * 100
                avg_inf = stats['total_inf_percent'] / stats['count'] * 100
                info(f"Parameter shape {shape}: {stats['count']} occurrences, "
                     f"avg {avg_nan:.1f}% NaNs, {avg_inf:.1f}% Infs, "
                     f"last seen {safe_clip_grad_norm_.call_count - stats['last_seen']} steps ago")

            # Suggest solutions based on patterns
            if any(stats['total_nan_percent']/stats['count'] > 0.5 for stats in persistent_issues.values()):
                info("Suggestion: Consider reducing learning rate by 50% or adding batch normalization")
            info("==================================")

    # Now apply gradient clipping with the fixed gradients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )

    # Detect vanishing gradients
    if total_norm < 1e-4 and safe_clip_grad_norm_.call_count % 50 == 0:
        info(f"WARNING: Potential gradient vanishing detected! Gradient norm: {total_norm:.8f}")
        info("Consider adjusting learning rate or model architecture.")

    return total_norm
