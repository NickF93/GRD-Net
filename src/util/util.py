"""
This module provides utility functions for managing TensorFlow sessions
and configuring GPU settings.

The module contains:
1. `clear_session`: A function to clear the current Keras session and free memory.
2. `config_gpu`: A function to configure GPUs by enabling memory growth to
   prevent out-of-memory (OOM) errors when running TensorFlow models.

These utilities are helpful when working with TensorFlow in environments where
memory management is critical, such as training large models or running multiple
GPU processes.

Usage:
    from utils import clear_session, config_gpu
    clear_session()
    config_gpu()
"""

import gc
from typing import List, Optional

import tensorflow as tf

def clear_session() -> None:
    """
    Clear the current Keras session and force garbage collection.

    This function is used to release memory when working with TensorFlow/Keras,
    particularly after finishing training or inference. It helps avoid memory
    leaks or accumulation of unused resources.

    Steps:
    1. Forces Python's garbage collector to release any unreferenced memory.
    2. Clears the current Keras backend session, releasing associated resources.
    3. Runs garbage collection again to free any further unreferenced memory.
    """
    # Collect garbage to free up memory before clearing the session
    gc.collect()
    
    # Clear the current TensorFlow/Keras session to free up resources
    tf.keras.backend.clear_session()
    
    # Collect garbage once again to ensure memory is completely freed
    gc.collect()

def config_gpu() -> None:
    """
    Configure GPU settings to manage memory growth and prevent OOM (Out Of Memory) errors.

    This function checks for available GPUs and sets memory growth to true, which
    ensures that TensorFlow allocates memory only as needed rather than pre-allocating
    the entire GPU memory. This can prevent out-of-memory errors when running multiple
    processes that use GPU resources.

    Steps:
    1. Retrieve a list of physical GPUs.
    2. For each GPU, enable memory growth to prevent TensorFlow from allocating
       all available memory at once.
    3. Print the number of physical and logical GPUs available for use.

    Raises
    ------
    RuntimeError
        If memory growth is set after GPUs have already been initialized, a runtime
        error is raised since memory growth must be set before initialization.
    """
    # List all physical devices of type 'GPU'
    gpus: List[Optional[tf.config.PhysicalDevice]] = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for each available GPU
            # This prevents TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # List logical devices (which TensorFlow sees after memory growth is set)
            logical_gpus: List[Optional[tf.config.LogicalDevice]] = tf.config.experimental.list_logical_devices('GPU')
            
            # Print the number of physical and logical GPUs available
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        
        except RuntimeError as e:
            # Memory growth must be set before any GPU is initialized
            # If the GPUs are already initialized, this error will be raised
            print(f"Error: {e}")
