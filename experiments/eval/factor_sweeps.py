"""
Factor sweep utilities for generating clean subspace visualizations.

Creates controlled sweeps of individual factors (rotation, translation, scale)
on a fixed set of anchor images, producing the Lie-group-style trajectory plots.
"""

import math
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def generate_rotation_sweep(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    n_angles: int = 36,
    angle_range: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate a sweep of rotations for anchor images.
    
    Args:
        model: The encoder model.
        images: Anchor images, shape (N, C, H, W).
        device: Computation device.
        n_angles: Number of rotation angles to sample.
        angle_range: (min, max) rotation in degrees.
    
    Returns:
        z_sweep: Encoded representations, shape (N * n_angles, d).
        angles: Array of rotation angles in degrees.
    """
    model.eval()
    
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)
    all_z = []
    
    with torch.no_grad():
        for angle in angles:
            # Apply rotation to all images
            rotated = torch.stack([
                TF.rotate(img, float(angle), fill=0) 
                for img in images
            ])
            rotated = rotated.to(device)
            
            # Get encoder output
            if hasattr(model, 'encoder'):
                z = model.encoder(rotated)
            else:
                z = model(rotated)
            
            all_z.append(z.cpu())
    
    # Stack: (n_angles, N, d) -> reshape for per-image trajectories
    z_sweep = torch.stack(all_z, dim=0)  # (n_angles, N, d)
    
    return z_sweep, angles


def generate_translation_sweep(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    axis: Literal["x", "y"] = "x",
    n_steps: int = 15,
    trans_range: Tuple[int, int] = (-7, 7),
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate a sweep of translations along one axis.
    
    Args:
        model: The encoder model.
        images: Anchor images, shape (N, C, H, W).
        device: Computation device.
        axis: 'x' or 'y' translation axis.
        n_steps: Number of translation steps.
        trans_range: (min, max) translation in pixels.
    
    Returns:
        z_sweep: Encoded representations, shape (n_steps, N, d).
        offsets: Array of translation offsets in pixels.
    """
    model.eval()
    
    offsets = np.linspace(trans_range[0], trans_range[1], n_steps).astype(int)
    all_z = []
    
    with torch.no_grad():
        for offset in offsets:
            if axis == "x":
                translate = [int(offset), 0]
            else:
                translate = [0, int(offset)]
            
            # Apply translation to all images
            translated = torch.stack([
                TF.affine(img, angle=0, translate=translate, scale=1.0, shear=0, fill=0)
                for img in images
            ])
            translated = translated.to(device)
            
            # Get encoder output
            if hasattr(model, 'encoder'):
                z = model.encoder(translated)
            else:
                z = model(translated)
            
            all_z.append(z.cpu())
    
    z_sweep = torch.stack(all_z, dim=0)  # (n_steps, N, d)
    
    return z_sweep, offsets


def generate_scale_sweep(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    n_steps: int = 11,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    pad_size: int = 40,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate a sweep of scales.
    
    Args:
        model: The encoder model.
        images: Anchor images, shape (N, C, H, W).
        device: Computation device.
        n_steps: Number of scale steps.
        scale_range: (min, max) scale factor.
        pad_size: Intermediate size for scaling operations.
    
    Returns:
        z_sweep: Encoded representations, shape (n_steps, N, d).
        scales: Array of scale factors.
    """
    model.eval()
    
    # Log-uniform scale sampling
    log_min = math.log(scale_range[0])
    log_max = math.log(scale_range[1])
    scales = np.exp(np.linspace(log_min, log_max, n_steps))
    
    orig_size = images.shape[-1]
    all_z = []
    
    with torch.no_grad():
        for scale in scales:
            scaled_list = []
            for img in images:
                # Pad
                pad_amount = (pad_size - orig_size) // 2
                if pad_amount > 0:
                    padded = TF.pad(img, pad_amount, fill=0)
                else:
                    padded = img
                
                # Scale
                new_size = int(pad_size * scale)
                scaled = TF.resize(padded, [new_size, new_size])
                
                # Center crop/pad back
                if new_size > pad_size:
                    scaled = TF.center_crop(scaled, pad_size)
                elif new_size < pad_size:
                    pad_needed = pad_size - new_size
                    pad_left = pad_needed // 2
                    pad_right = pad_needed - pad_left
                    scaled = TF.pad(scaled, [pad_left, pad_left, pad_right, pad_right], fill=0)
                
                # Final crop to original size
                scaled = TF.center_crop(scaled, orig_size)
                scaled_list.append(scaled)
            
            scaled_batch = torch.stack(scaled_list).to(device)
            
            # Get encoder output
            if hasattr(model, 'encoder'):
                z = model.encoder(scaled_batch)
            else:
                z = model(scaled_batch)
            
            all_z.append(z.cpu())
    
    z_sweep = torch.stack(all_z, dim=0)  # (n_steps, N, d)
    
    return z_sweep, scales


def compute_sweep_delta_z(
    z_sweep: torch.Tensor,
    reference_idx: int = 0,
) -> torch.Tensor:
    """
    Compute delta_z relative to a reference point in the sweep.
    
    Args:
        z_sweep: Shape (n_steps, N, d).
        reference_idx: Index of reference point (usually center or identity).
    
    Returns:
        delta_z: Shape (n_steps * N, d).
    """
    n_steps, N, d = z_sweep.shape
    z_ref = z_sweep[reference_idx]  # (N, d)
    
    # Compute delta relative to reference for each anchor
    delta_z = z_sweep - z_ref.unsqueeze(0)  # (n_steps, N, d)
    
    # Flatten to (n_steps * N, d)
    return delta_z.reshape(-1, d)


def run_full_factor_sweep(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    factors: List[str] = ["rot", "trans_x", "trans_y", "scale"],
) -> Dict[str, Dict]:
    """
    Run sweeps for all factors and return results.
    
    Args:
        model: The encoder model.
        images: Anchor images, shape (N, C, H, W).
        device: Computation device.
        factors: List of factors to sweep.
    
    Returns:
        Dict mapping factor name to {"z_sweep": tensor, "params": array, "delta_z": tensor}.
    """
    results = {}
    
    # Find identity/center index for reference
    center_idx = None
    
    for factor in factors:
        try:
            if factor == "rot":
                z_sweep, params = generate_rotation_sweep(model, images, device)
                # Identity is 0 degrees
                center_idx = len(params) // 2
            elif factor == "trans_x":
                z_sweep, params = generate_translation_sweep(model, images, device, axis="x")
                center_idx = len(params) // 2
            elif factor == "trans_y":
                z_sweep, params = generate_translation_sweep(model, images, device, axis="y")
                center_idx = len(params) // 2
            elif factor == "scale":
                z_sweep, params = generate_scale_sweep(model, images, device)
                # Identity is scale=1.0, find closest
                center_idx = np.argmin(np.abs(params - 1.0))
            else:
                continue
            
            delta_z = compute_sweep_delta_z(z_sweep, reference_idx=center_idx)
            
            results[factor] = {
                "z_sweep": z_sweep,
                "params": params,
                "delta_z": delta_z,
            }
        except Exception as e:
            print(f"Warning: Factor sweep {factor} failed: {e}")
    
    return results

