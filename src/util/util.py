"""
This module provides utility functions for managing TensorFlow sessions
and configuring GPU settings.

The module contains:
1. `clear_session`: Clears the current Keras session and frees memory.
2. `config_gpu`: Configures GPU settings to prevent out-of-memory errors.
3. `set_seed`: Sets a deterministic seed for reproducibility.
4. `model_logger`: Logs and saves visual representations of TensorFlow models.

Usage:
    from this_module import clear_session, config_gpu, set_seed, model_logger
    clear_session()
    config_gpu()
    seed = set_seed(42)
    model_logger(model)
"""

import os
import time
import random
import math
import gc
import logging
from typing import Optional, List

import tensorflow as tf
import numpy as np
import colorama

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

def set_seed(seed: Optional[int] = None) -> int:
    """
    Sets the random seeds for necessary libraries to ensure reproducibility.

    This function sets the random seeds for the `random`, `numpy`, and `tensorflow`
    modules to ensure that experiments involving random processes can be replicated
    exactly. If no seed is provided, the function generates a seed based on the current
    time, adjusted to avoid a seed value of zero.

    Parameters:
    - seed (Optional[int]): The seed value to use for random number generation.
      If `None`, a seed will be automatically generated based on the current time.

    Returns:
    - int: The seed used to set the random number generators.

    Note:
    - The generated seed from the current time is computed to avoid being zero,
      which is important as some random number generators behave differently with a
      zero seed.
    """
    if seed is None:
        t = time.time()
        a, b = math.modf(t)
        a = float(int(a * (10 ** 7)))
        if a == 0:
            a = 1
        seed = int((b / a) * 1000)
    else:
        seed = int(seed)

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    return seed

class LevelNameFormatter(logging.Formatter):
    """
    A custom formatter for logging that applies color coding to log level names.

    Attributes:
    - level_colors (dict): A dictionary mapping log level names to their respective
      color codes using ANSI escape sequences provided by colorama.

    Methods:
    - format: Overrides the base class method to insert color codes into the log messages.
    """

    level_colors = {
        "DEBUG": colorama.Fore.BLUE,
        "INFO": colorama.Fore.GREEN,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as text.

        Inserts ANSI color escape sequences around the log level name based on the
        severity of the log record. This method is called automatically by the logging
        library when a message is logged.

        Parameters:
        - record (logging.LogRecord): The log record to be formatted.

        Returns:
        - str: A formatted string with color-coded log level names.
        """
        log_fmt = f"{colorama.Fore.RESET}{self.level_colors[record.levelname]}%(levelname)s{colorama.Fore.RESET}: %(asctime)s - %(name)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def model_logger(model: tf.keras.models.Model, logger: Optional[logging.Logger] = None, 
                 save_path: Optional[str] = None, print_visualkeras: bool = False) -> None:
    """
    Logs the TensorFlow model summary and saves visualizations of the model.

    Parameters:
    - model (tf.keras.models.Model): The model to log and visualize.
    - logger (Optional[logging.Logger]): Logger to use for outputting the summary and other info.
    - save_path (Optional[str]): Path where the model visualizations should be saved.
    - print_visualkeras (bool): Flag to indicate whether to use visualkeras for additional visualization.

    This function prints the model's summary either to the standard output or to a provided logger.
    It also saves a graphical plot of the model to the specified directory if it exists. Optionally,
    it can generate and save a visualization using visualkeras if that package is available.
    """
    # Print the model summary using the provided logger or default print function.
    if logger is None:
        model.summary()
    else:
        model.summary(print_fn=logger.debug)

    # Check if save path is valid and save the model plot.
    if save_path and os.path.exists(save_path) and os.path.isdir(save_path):
        plot_file_path = os.path.join(save_path, f'{model.name}_model_plot.png')
        tf.keras.utils.plot_model(
            model,
            to_file=plot_file_path,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=250,
            show_layer_activations=True,
            show_trainable=True
        )
        if logger:
            logger.info(f"Model plot saved to {plot_file_path}")

        # Generate a visualkeras model plot if requested and possible.
        if print_visualkeras:
            try:
                import visualkeras

                vk_file_path = os.path.join(save_path, f'{model.name}_visualkeras.png')
                visualkeras.layered_view(model, to_file=vk_file_path)
                if logger:
                    logger.info(f"Visualkeras model visualization saved to {vk_file_path}")

            except ImportError:
                if logger:
                    logger.warning("visualkeras is not installed. Please install it to enable visualkeras plotting.")
            except Exception as ex:
                if logger:
                    logger.warning(f"Error plotting with visualkeras: {ex}")
