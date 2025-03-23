import torch
import logging
import os
import platform
from typing import Optional, Dict, Union, Tuple

class DeviceManager:
    """
    Centralized manager for device-related operations throughout the project.
    Handles detection, configuration, and optimization for various hardware accelerators.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for DeviceManager"""
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, memory_fraction: float = 0.9, verbose: bool = True):
        """Initialize device manager if not already initialized"""
        if self._initialized:
            return
            
        self.memory_fraction = memory_fraction
        self.verbose = verbose
        self._device = None
        self._device_info = {}
        self._initialized = True
        self._init_device()
    
    def _init_device(self):
        """Initialize the device and apply optimizations"""
        # Detect available hardware
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._device_info = {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "compute_capability": torch.cuda.get_device_capability(0)
            }
            
            # Apply memory optimizations
            if self.memory_fraction < 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                    if self.verbose:
                        logging.info(f"Set CUDA memory fraction to {self.memory_fraction}")
                except:
                    logging.warning("Failed to set CUDA memory fraction")
            
            # Set cudnn optimizations
            torch.backends.cudnn.benchmark = True
            
            if self.verbose:
                logging.info(f"Using CUDA device: {self._device_info['name']}")
                
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._device_info = {
                "type": "mps",
                "name": f"Apple Silicon ({platform.processor()})",
                "count": 1
            }
            
            if self.verbose:
                logging.info(f"Using MPS device (Apple Silicon)")
                
        else:
            self._device = torch.device("cpu")
            self._device_info = {
                "type": "cpu",
                "name": f"CPU ({platform.processor()})",
                "count": os.cpu_count()
            }
            
            if self.verbose:
                logging.info(f"Using CPU device with {os.cpu_count()} cores")
    
    @property
    def device(self) -> torch.device:
        """Get the PyTorch device object"""
        return self._device
    
    @property
    def is_cuda(self) -> bool:
        """Check if CUDA is available and being used"""
        return self._device.type == "cuda"
    
    @property
    def is_mps(self) -> bool:
        """Check if MPS (Apple Silicon) is available and being used"""
        return self._device.type == "mps"
    
    @property
    def is_cpu(self) -> bool:
        """Check if CPU is being used"""
        return self._device.type == "cpu"
        
    @property
    def is_gpu(self) -> bool:
        """Check if any GPU (CUDA or MPS) is being used"""
        return self.is_cuda or self.is_mps
    
    def get_info(self) -> Dict[str, Union[str, int, float]]:
        """Get detailed information about the current device"""
        info = dict(self._device_info)
        
        # Add memory info for CUDA devices
        if self.is_cuda:
            info["memory_allocated"] = torch.cuda.memory_allocated()
            info["memory_reserved"] = torch.cuda.memory_reserved()
            info["max_memory_allocated"] = torch.cuda.max_memory_allocated()
            
            # Calculate available memory
            if "memory_total" in info:
                info["memory_available"] = info["memory_total"] - info["memory_reserved"]
                
            # Format memory values as GB for human-readable output
            for key in ["memory_total", "memory_allocated", "memory_reserved", "memory_available"]:
                if key in info:
                    info[f"{key}_gb"] = info[key] / (1024**3)
        
        return info
    
    def clear_cache(self):
        """Clear device memory cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            if self.verbose:
                logging.info("Cleared CUDA cache")
        elif self.is_mps:
            # MPS doesn't have an explicit cache clearing mechanism yet
            pass
    
    def optimize_for_inference(self):
        """Apply optimizations for inference workloads"""
        if self.is_cuda:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.verbose:
                logging.info("Applied CUDA optimizations for inference")
    
    def optimize_for_training(self):
        """Apply optimizations for training workloads"""
        if self.is_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if self.verbose:
                logging.info("Applied CUDA optimizations for training")

    def tensor_to_device(self, tensor):
        """Move a tensor to the current device with optimal settings"""
        if tensor is None:
            return None
        
        if not isinstance(tensor, torch.Tensor):
            # Convert numpy arrays or lists to tensors
            tensor = torch.tensor(tensor)
            
        # Use non_blocking for potentially faster transfers with pinned memory
        return tensor.to(self.device, non_blocking=True)
    
    def print_summary(self):
        """Print a detailed summary of the device configuration"""
        info = self.get_info()
        
        print("\n" + "="*50)
        print(f"Device Summary:")
        print(f"  - Type: {info['type']}")
        print(f"  - Name: {info['name']}")
        
        if self.is_gpu:
            print(f"  - Count: {info['count']}")
            
            if self.is_cuda:
                print(f"  - Compute Capability: {info['compute_capability'][0]}.{info['compute_capability'][1]}")
                print(f"  - Memory Total: {info.get('memory_total_gb', 0):.2f} GB")
                print(f"  - Memory Allocated: {info.get('memory_allocated_gb', 0):.2f} GB")
                print(f"  - Memory Reserved: {info.get('memory_reserved_gb', 0):.2f} GB")
                print(f"  - Memory Available: {info.get('memory_available_gb', 0):.2f} GB")
                print(f"  - Memory Fraction: {self.memory_fraction:.2f}")
        else:
            print(f"  - CPU Cores: {info.get('count', 'Unknown')}")
            
        print("="*50 + "\n")


# Singleton instance for global use
_device_manager = None

def get_device_manager(memory_fraction: float = 0.9, verbose: bool = True) -> DeviceManager:
    """Get the global DeviceManager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(memory_fraction=memory_fraction, verbose=verbose)
    return _device_manager

def get_device() -> torch.device:
    """
    Get the optimal device for the current hardware configuration.
    This is a backward-compatible function that works like the original get_device().
    """
    return get_device_manager().device

def is_cuda_available() -> bool:
    """Check if CUDA is available and being used"""
    return get_device_manager().is_cuda

def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available and being used"""
    return get_device_manager().is_mps

def is_gpu_available() -> bool:
    """Check if any GPU acceleration is available and being used"""
    return get_device_manager().is_gpu

def clear_gpu_cache():
    """Clear GPU memory cache"""
    get_device_manager().clear_cache()

def to_device(tensor):
    """Move a tensor to the optimal device"""
    return get_device_manager().tensor_to_device(tensor)

def print_device_info():
    """Print detailed information about the current device"""
    get_device_manager().print_summary()

# Simple function to get device memory info (for CUDA devices)
def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in GB"""
    manager = get_device_manager()
    if not manager.is_cuda:
        return {"error": "No CUDA device available"}
    
    info = manager.get_info()
    return {
        "total_gb": info.get("memory_total_gb", 0),
        "allocated_gb": info.get("memory_allocated_gb", 0),
        "reserved_gb": info.get("memory_reserved_gb", 0),
        "available_gb": info.get("memory_available_gb", 0)
    }

# For testing
if __name__ == "__main__":
    # Test the device manager
    print_device_info()
    
    # Basic example of using the device utilities
    device = get_device()
    print(f"Using device: {device}")
    
    # Create a test tensor
    x = torch.randn(100, 100)
    x_device = to_device(x)
    print(f"Tensor moved to {x_device.device}")
    
    # Show memory usage if on CUDA
    if is_cuda_available():
        print("CUDA Memory Information:")
        memory_info = get_gpu_memory_info()
        for key, value in memory_info.items():
            print(f"  {key}: {value:.2f} GB")