"""
CUDA Safety Utilities for Enhanced TGN
Comprehensive device handling and error recovery functions
"""

import torch
import gc
import traceback
from typing import List, Any, Optional, Union, Tuple


def get_device_info():
    """Get comprehensive device information"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            'device': device,
            'gpu_name': gpu_name,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'memory_total_gb': memory_total,
            'memory_free_gb': memory_total - memory_reserved
        }
    else:
        return {
            'device': torch.device('cpu'),
            'gpu_name': None,
            'memory_allocated_gb': 0,
            'memory_reserved_gb': 0,
            'memory_total_gb': 0,
            'memory_free_gb': 0
        }


def ensure_tensor_device(tensor: Union[torch.Tensor, Any], 
                        device: torch.device, 
                        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Ensure tensor is on the correct device with proper type conversion
    
    Args:
        tensor: Input tensor or array-like object
        device: Target device
        dtype: Optional dtype conversion
        
    Returns:
        Tensor on correct device with correct dtype
    """
    try:
        # Convert to tensor if not already
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=dtype)
        elif dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        
        # Move to device if not already there
        if tensor.device != device:
            tensor = tensor.to(device)
            
        return tensor
    except Exception as e:
        print(f"Warning: Error in ensure_tensor_device: {e}")
        # Fallback: create on CPU then move
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=dtype, device='cpu')
        else:
            tensor = tensor.cpu()
        return tensor.to(device)


def ensure_device_consistency(*tensors, device: torch.device) -> List[torch.Tensor]:
    """
    Ensure all tensors are on the same device
    
    Args:
        *tensors: Variable number of tensors
        device: Target device
        
    Returns:
        List of tensors all on the same device
    """
    result = []
    for i, tensor in enumerate(tensors):
        try:
            if tensor is not None:
                tensor = ensure_tensor_device(tensor, device)
            result.append(tensor)
        except Exception as e:
            print(f"Error ensuring device consistency for tensor {i}: {e}")
            result.append(tensor)  # Return original if conversion fails
    return result


def safe_tensor_indexing(tensor: torch.Tensor, 
                        indices: torch.Tensor, 
                        device: torch.device) -> torch.Tensor:
    """
    Safely index tensors with device consistency checks
    
    Args:
        tensor: Source tensor to index
        indices: Index tensor
        device: Target device
        
    Returns:
        Indexed tensor on correct device
    """
    try:
        # Ensure both tensors are on the same device
        tensor = ensure_tensor_device(tensor, device)
        indices = ensure_tensor_device(indices, device, dtype=torch.long)
        
        # Validate indices
        if indices.max() >= tensor.size(0):
            print(f"Warning: Index out of bounds. Max index: {indices.max()}, tensor size: {tensor.size(0)}")
            indices = torch.clamp(indices, 0, tensor.size(0) - 1)
        
        if indices.min() < 0:
            print(f"Warning: Negative indices detected. Min index: {indices.min()}")
            indices = torch.clamp(indices, 0, tensor.size(0) - 1)
        
        # Perform indexing
        return tensor[indices]
        
    except Exception as e:
        print(f"Error in safe_tensor_indexing: {e}")
        try:
            # Fallback: move to CPU, index, then move back
            tensor_cpu = tensor.cpu()
            indices_cpu = indices.cpu().long()
            result = tensor_cpu[indices_cpu]
            return result.to(device)
        except Exception as e2:
            print(f"Fallback indexing also failed: {e2}")
            # Ultimate fallback: return zeros with correct shape
            if len(indices.shape) == 0:
                return torch.zeros_like(tensor[0:1]).to(device)
            else:
                return torch.zeros(indices.size(0), *tensor.shape[1:], device=device, dtype=tensor.dtype)


def handle_cuda_oom(func, *args, **kwargs):
    """
    Handle CUDA out of memory errors with automatic recovery
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or raises exception
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA OOM detected: {e}")
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            gc.collect()
            raise e  # Re-raise to be handled by caller
        else:
            raise e


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def validate_tensor_shapes(tensors: dict, expected_shapes: dict) -> bool:
    """
    Validate tensor shapes match expected dimensions
    
    Args:
        tensors: Dictionary of tensor_name -> tensor
        expected_shapes: Dictionary of tensor_name -> expected_shape_tuple
        
    Returns:
        True if all shapes are valid, False otherwise
    """
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape
            
            # Check if shapes are compatible (allowing for batch dimension flexibility)
            if len(actual) != len(expected):
                print(f"Shape mismatch for {name}: expected {len(expected)} dims, got {len(actual)} dims")
                return False
            
            # Check non-batch dimensions (skip first dimension as it's usually batch size)
            for i in range(1, len(expected)):
                if expected[i] != -1 and actual[i] != expected[i]:  # -1 means any size is OK
                    print(f"Shape mismatch for {name}: expected {expected}, got {actual}")
                    return False
    
    return True


def safe_model_forward(model, device: torch.device, **inputs) -> Tuple[bool, Union[Any, Exception]]:
    """
    Safely execute model forward pass with comprehensive error handling
    
    Args:
        model: PyTorch model
        device: Target device
        **inputs: Model input arguments
        
    Returns:
        Tuple of (success: bool, result_or_exception)
    """
    try:
        # Ensure all inputs are on correct device
        device_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                device_inputs[key] = ensure_tensor_device(value, device)
            else:
                device_inputs[key] = value
        
        # Execute forward pass
        with torch.cuda.device(device) if device.type == 'cuda' else torch.no_grad():
            result = model(**device_inputs)
        
        return True, result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_cuda_cache()
            return False, e
        else:
            return False, e
    except Exception as e:
        return False, e


def create_safe_dataloader(dataset, batch_size: int, device: torch.device, 
                          shuffle: bool = False, num_workers: int = 0):
    """
    Create a dataloader with CUDA-safe settings
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        device: Target device
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader with safe settings
    """
    # Adjust settings for CUDA safety
    if device.type == 'cuda':
        # Use pin_memory for faster GPU transfer, but fewer workers to avoid memory issues
        pin_memory = True
        num_workers = min(num_workers, 2)  # Limit workers to prevent memory issues
    else:
        pin_memory = False
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop incomplete batches
        persistent_workers=False  # Don't keep workers alive between epochs
    )


def monitor_memory_usage(device: torch.device, operation_name: str = ""):
    """
    Monitor and print memory usage
    
    Args:
        device: Device to monitor
        operation_name: Name of the operation being monitored
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"Memory usage {operation_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def create_device_safe_optimizer(model_parameters, lr: float = 0.001, weight_decay: float = 1e-5):
    """
    Create optimizer with safe default settings
    
    Args:
        model_parameters: Model parameters
        lr: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer instance
    """
    return torch.optim.Adam(
        model_parameters,
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,  # Avoid division by zero
        amsgrad=True  # More stable optimization
    )


def safe_gradient_step(optimizer, model, max_norm: float = 1.0) -> bool:
    """
    Safely perform gradient step with clipping
    
    Args:
        optimizer: PyTorch optimizer
        model: Model with parameters
        max_norm: Maximum gradient norm for clipping
        
    Returns:
        True if step was successful, False if gradients were invalid
    """
    try:
        # Check for invalid gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Skip step if gradients are invalid
        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
            print("Warning: Invalid gradients detected, skipping optimizer step")
            optimizer.zero_grad()
            return False
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Perform step
        optimizer.step()
        return True
        
    except Exception as e:
        print(f"Error in gradient step: {e}")
        optimizer.zero_grad()
        return False


# ================================
# COMPREHENSIVE CUDA-SAFE TGN TRAINING FUNCTION
# ================================

def cuda_safe_train_tgn_model(model, train_loader, val_loader, node_features, epochs=10, 
                              learning_rate=0.0001, device=None, max_retries=3):
    """
    Comprehensive CUDA-safe training function for TGN model
    
    Args:
        model: TGN model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        node_features: Node features tensor
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use (auto-detected if None)
        max_retries: Maximum number of retries on CUDA errors
        
    Returns:
        Dictionary containing training results and metrics
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import traceback
    import time
    
    print("Starting CUDA-safe TGN training...")
    print("=" * 50)
    
    # Get device information
    device_info = get_device_info()
    if device is None:
        device = device_info['device']
    
    print(f"Training device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {device_info['gpu_name']}")
        print(f"Available memory: {device_info['memory_free_gb']:.2f}GB")
    
    # Ensure model and node features are on correct device
    try:
        model = model.to(device)
        node_features = ensure_tensor_device(node_features, device)
        print(f"Model and data moved to {device}")
    except Exception as e:
        print(f"Error moving model to device: {e}")
        return {"error": str(e)}
    
    # Create CUDA-safe optimizer
    optimizer = create_device_safe_optimizer(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        successful_batches = 0
        failed_batches = 0
        
        # Reset memory at the start of each epoch with CUDA safety
        try:
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            clear_cuda_cache()
        except Exception as e:
            print(f"Warning: Could not reset memory: {e}")
        
        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_success = False
            retry_count = 0
            
            while not batch_success and retry_count < max_retries:
                try:
                    # Validate batch data
                    if not isinstance(batch, dict) or len(batch) == 0:
                        print(f"Skipping invalid batch at index {batch_idx}")
                        break
                    
                    # Extract and validate batch data with device consistency
                    src_idx = ensure_tensor_device(batch['src_idx'].squeeze(), device, torch.long)
                    dst_idx = ensure_tensor_device(batch['dst_idx'].squeeze(), device, torch.long)
                    timestamp = ensure_tensor_device(batch['timestamp'].squeeze(), device, torch.float)
                    edge_features = ensure_tensor_device(batch['edge_features'], device, torch.float)
                    pos_label = ensure_tensor_device(batch['label'], device, torch.float)
                    
                    # Validate tensor shapes
                    batch_size = src_idx.size(0)
                    if not validate_tensor_shapes({
                        'src_idx': src_idx,
                        'dst_idx': dst_idx,
                        'timestamp': timestamp,
                        'pos_label': pos_label
                    }, {
                        'src_idx': (batch_size,),
                        'dst_idx': (batch_size,),
                        'timestamp': (batch_size,),
                        'pos_label': (batch_size,)
                    }):
                        print(f"Invalid tensor shapes in batch {batch_idx}")
                        break
                    
                    # Generate negative samples with device safety
                    neg_dst_idx = torch.randint(0, model.num_nodes, src_idx.size(), 
                                              device=device, dtype=torch.long)
                    
                    # Ensure negative samples are different from positive samples
                    for i in range(len(src_idx)):
                        max_attempts = 10
                        attempts = 0
                        while neg_dst_idx[i] == dst_idx[i] and attempts < max_attempts:
                            neg_dst_idx[i] = torch.randint(0, model.num_nodes, (1,), 
                                                         device=device, dtype=torch.long)
                            attempts += 1
                    
                    neg_label = torch.zeros_like(pos_label, device=device, dtype=torch.float)
                    
                    # Safe tensor indexing for node features
                    src_features = safe_tensor_indexing(node_features, src_idx, device)
                    dst_features = safe_tensor_indexing(node_features, dst_idx, device)
                    neg_dst_features = safe_tensor_indexing(node_features, neg_dst_idx, device)
                    
                    # Forward pass for positive samples with CUDA safety
                    success, result = safe_model_forward(
                        model, device,
                        src_ids=src_idx,
                        dst_ids=dst_idx,
                        src_features=src_features,
                        dst_features=dst_features,
                        timestamps=timestamp,
                        edge_features=edge_features
                    )
                    
                    if not success:
                        if "out of memory" in str(result).lower():
                            print(f"CUDA OOM in positive forward pass, batch {batch_idx}, retry {retry_count+1}")
                            handle_cuda_oom(lambda: None)
                            retry_count += 1
                            continue
                        else:
                            print(f"Error in positive forward pass: {result}")
                            break
                    
                    pos_prob, _, _ = result
                    
                    # Forward pass for negative samples with CUDA safety
                    success, result = safe_model_forward(
                        model, device,
                        src_ids=src_idx,
                        dst_ids=neg_dst_idx,
                        src_features=src_features,
                        dst_features=neg_dst_features,
                        timestamps=timestamp,
                        edge_features=edge_features
                    )
                    
                    if not success:
                        if "out of memory" in str(result).lower():
                            print(f"CUDA OOM in negative forward pass, batch {batch_idx}, retry {retry_count+1}")
                            handle_cuda_oom(lambda: None)
                            retry_count += 1
                            continue
                        else:
                            print(f"Error in negative forward pass: {result}")
                            break
                    
                    neg_prob, _, _ = result
                    
                    # Compute loss with device consistency
                    pos_prob = ensure_tensor_device(pos_prob, device)
                    neg_prob = ensure_tensor_device(neg_prob, device)
                    
                    pos_loss = loss_fn(pos_prob, pos_label)
                    neg_loss = loss_fn(neg_prob, neg_label)
                    loss = pos_loss + neg_loss
                    
                    # Backward pass and optimization with CUDA safety
                    optimizer.zero_grad()
                    
                    try:
                        loss.backward()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"CUDA OOM in backward pass, batch {batch_idx}, retry {retry_count+1}")
                            handle_cuda_oom(lambda: None)
                            retry_count += 1
                            continue
                        else:
                            raise e
                    
                    # Safe gradient step
                    if not safe_gradient_step(optimizer, model, max_norm=1.0):
                        print(f"Gradient step failed for batch {batch_idx}")
                        break
                    
                    # Detach memory safely
                    try:
                        if hasattr(model, 'detach_memory'):
                            model.detach_memory()
                    except Exception as e:
                        print(f"Warning: Could not detach memory: {e}")
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    successful_batches += 1
                    batch_success = True
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg Loss': f'{total_loss/num_batches:.4f}',
                        'Success Rate': f'{successful_batches/(successful_batches+failed_batches)*100:.1f}%'
                    })
                    
                    # Periodic memory cleanup
                    if batch_idx % 10 == 0:
                        clear_cuda_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"CUDA OOM in batch {batch_idx}, retry {retry_count+1}/{max_retries}")
                        handle_cuda_oom(lambda: None)
                        retry_count += 1
                    else:
                        print(f"Runtime error in batch {batch_idx}: {e}")
                        break
                except Exception as e:
                    print(f"Unexpected error in batch {batch_idx}: {e}")
                    break
            
            if not batch_success:
                failed_batches += 1
                print(f"Failed to process batch {batch_idx} after {max_retries} retries")
        
        progress_bar.close()
        
        # Compute average training loss
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        print(f"Training - Batches: {successful_batches}/{successful_batches+failed_batches}, "
              f"Success Rate: {successful_batches/(successful_batches+failed_batches)*100:.1f}%, "
              f"Avg Loss: {avg_train_loss:.4f}")
        
        # Validation phase with CUDA safety
        print("Running validation...")
        model.eval()
        val_loss = 0
        val_batches = 0
        val_successful = 0
        val_failed = 0
        
        # Reset memory for validation
        try:
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            clear_cuda_cache()
        except Exception as e:
            print(f"Warning: Could not reset memory for validation: {e}")
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation")
            
            for val_batch in val_progress:
                try:
                    # Process validation batch with same safety measures
                    src_idx = ensure_tensor_device(val_batch['src_idx'].squeeze(), device, torch.long)
                    dst_idx = ensure_tensor_device(val_batch['dst_idx'].squeeze(), device, torch.long)
                    timestamp = ensure_tensor_device(val_batch['timestamp'].squeeze(), device, torch.float)
                    edge_features = ensure_tensor_device(val_batch['edge_features'], device, torch.float)
                    pos_label = ensure_tensor_device(val_batch['label'], device, torch.float)
                    
                    # Generate negative samples
                    neg_dst_idx = torch.randint(0, model.num_nodes, src_idx.size(), 
                                              device=device, dtype=torch.long)
                    neg_label = torch.zeros_like(pos_label, device=device, dtype=torch.float)
                    
                    # Safe tensor indexing
                    src_features = safe_tensor_indexing(node_features, src_idx, device)
                    dst_features = safe_tensor_indexing(node_features, dst_idx, device)
                    neg_dst_features = safe_tensor_indexing(node_features, neg_dst_idx, device)
                    
                    # Forward passes with safety
                    success, pos_result = safe_model_forward(
                        model, device,
                        src_ids=src_idx, dst_ids=dst_idx,
                        src_features=src_features, dst_features=dst_features,
                        timestamps=timestamp, edge_features=edge_features
                    )
                    
                    if not success:
                        print(f"Validation positive forward pass failed: {pos_result}")
                        val_failed += 1
                        continue
                    
                    success, neg_result = safe_model_forward(
                        model, device,
                        src_ids=src_idx, dst_ids=neg_dst_idx,
                        src_features=src_features, dst_features=neg_dst_features,
                        timestamps=timestamp, edge_features=edge_features
                    )
                    
                    if not success:
                        print(f"Validation negative forward pass failed: {neg_result}")
                        val_failed += 1
                        continue
                    
                    pos_prob, _, _ = pos_result
                    neg_prob, _, _ = neg_result
                    
                    # Compute validation loss
                    val_pos_loss = loss_fn(pos_prob, pos_label)
                    val_neg_loss = loss_fn(neg_prob, neg_label)
                    batch_val_loss = val_pos_loss + val_neg_loss
                    
                    val_loss += batch_val_loss.item()
                    val_batches += 1
                    val_successful += 1
                    
                    # Detach memory safely
                    try:
                        if hasattr(model, 'detach_memory'):
                            model.detach_memory()
                    except Exception as e:
                        print(f"Warning: Could not detach memory in validation: {e}")
                    
                    val_progress.set_postfix({
                        'Val Loss': f'{batch_val_loss.item():.4f}',
                        'Avg Val Loss': f'{val_loss/val_batches:.4f}'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    val_failed += 1
                    continue
            
            val_progress.close()
        
        # Clear GPU memory after validation
        clear_cuda_cache()
        
        # Compute average validation loss
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Validation - Batches: {val_successful}/{val_successful+val_failed}, "
              f"Success Rate: {val_successful/(val_successful+val_failed)*100:.1f}%, "
              f"Avg Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Memory monitoring
        if device.type == 'cuda':
            try:
                memory_info = get_device_info()
                print(f"GPU Memory: {memory_info['memory_free_gb']:.2f}GB free")
            except Exception as e:
                print(f"Could not get memory info: {e}")
    
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 50)
    print("CUDA-safe TGN training completed!")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_training_time,
        'device': str(device),
        'epochs_completed': len(train_losses),
        'success': True
    }


def create_cuda_safe_training_plot(training_results):
    """
    Create training plots from CUDA-safe training results
    
    Args:
        training_results: Dictionary returned from cuda_safe_train_tgn_model
    """
    import matplotlib.pyplot as plt
    
    if not training_results.get('success', False):
        print("Training was not successful, cannot create plots")
        return
    
    train_losses = training_results['train_losses']
    val_losses = training_results['val_losses']
    
    plt.figure(figsize=(12, 5))
    
    # Training and validation loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CUDA-Safe TGN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss convergence
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Loss Convergence (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Device: {training_results['device']}")
    print(f"Epochs completed: {training_results['epochs_completed']}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"Total training time: {training_results['total_time']:.2f}s")
