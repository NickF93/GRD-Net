import os
import tempfile
from typing import Dict, Tuple, List, Type, Any

import requests
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
import matplotlib.pyplot as plt
import numpy as np


class ExperimentManager:
    """
    This class manages the logging of experiments to TensorBoard and MLflow. It handles:
    - TensorBoard writer setup
    - Checking if the MLflow server is available
    - Starting and stopping MLflow runs
    - Logging metrics, parameters, images, and saving models to both TensorBoard and MLflow
    """

    def __init__(self, mlflow_uri: str, experiment_name: str, tensorboard_logdir: str, mlflow_alt_logdir: str) -> None:
        """
        Initialize the ExperimentManager with MLflow server URI, experiment name, and TensorBoard log directory.

        Args:
            mlflow_uri (str): URI of the MLflow tracking server.
            experiment_name (str): Name of the MLflow experiment.
            tensorboard_logdir (str): Directory to save TensorBoard logs.
        """
        self._mlflow_uri: str = mlflow_uri
        self._experiment_name: str = experiment_name
        self._tensorboard_logdir: str = tensorboard_logdir
        self._mlflow_alt_logdir: str = mlflow_alt_logdir
        self._run = None
        self.writer = None

    def __enter__(self) -> "ExperimentManager":
        """
        Context manager entry method. Sets up the experiment by initializing the TensorBoard callback and
        connecting to the MLflow server (if available).
        
        Returns:
            ExperimentManager: The initialized experiment manager instance.
        """
        self._check_mlflow_server()
        self._setup_experiment()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Context manager exit method. Ends the MLflow run (if active).
        Cleans up resources after the experiment.
        """
        self._close_experiment()

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to both TensorBoard and MLflow.

        Args:
            metrics (Dict[str, Any]): Dictionary containing metric names and their values.
            step (int): Current step or epoch in the experiment.
        """
        for metric_name, value in metrics.items():
            # Log to TensorBoard
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar(metric_name, value, step=step)

            # Log to MLflow
            mlflow.log_metric(metric_name, value, step)

        print(f"Metrics logged for step {step}.")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params (Dict[str, Any]): Dictionary containing parameter names and their values.
        """
        for param_name, value in params.items():
            mlflow.log_param(param_name, value)
        print(f"Parameters logged: {params}")

    def log_image(self, image: np.ndarray, tag: str, step: int, image_name: str = "image.png") -> None:
        """
        Log an image to both TensorBoard and MLflow (as an artifact).

        Args:
            image (np.ndarray): The image to log.
            tag (str): TensorBoard tag for the image.
            step (int): The step or epoch number.
            image_name (str): Name of the image file to be saved as an artifact in MLflow.
        """
        # Log image to TensorBoard

        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.image(tag, np.expand_dims(image, 0), step=step)
        print(f"Image logged to TensorBoard with tag '{tag}'.")

        # Save the image locally and log it to MLflow as an artifact
        plt.imsave(image_name, image)
        mlflow.log_artifact(image_name)
        print(f"Image logged to MLflow as artifact '{image_name}'.")
        os.remove(image_name)  # Clean up the saved file

    def save_model(self,
                model: keras.Model,
                model_name: str, 
                input_spec: Dict[str, Tuple[Tuple[int], Dict[int, str]]], 
                outputs: Dict[str, Tuple[int]], 
                model_params: Dict[str, Any],  
                save_tf_format: bool = True,
                save_h5_format: bool = False) -> None:
        """
        Save the model in TensorFlow SavedModel and H5 formats. Log the models to MLflow.
        Input tensors are created based on input_spec, and the model is saved in the requested formats.
        Temporary files are saved to the system's temp directory.
        
        Args:
            model_name (str): The name to give to the saved model files.
            input_spec (Dict[str, Tuple[Tuple[int], Dict[int, str]]]): 
                Dictionary where key is input name, and value is a tuple containing:
                - A tuple for the input shape (batch, channels, height, width).
                - A dictionary specifying dynamic axes (e.g., batch_size).
            outputs (Dict[str, Tuple[int]]): A dictionary containing output shapes.
            model_params (Dict[str, Any]): A dictionary containing parameters to instantiate the model class.
            save_tf_format (bool, optional): Whether to save in TensorFlow format. Defaults to True.
            save_h5_format (bool, optional): Whether to save in H5 format. Defaults to False.
        """
        tmp_dir = tempfile.gettempdir()

        if save_tf_format:
            tf_model_path = os.path.join(tmp_dir, f"{model_name}")
            model.save(tf_model_path)
            mlflow.tensorflow.log_model(tf_model_path, "model_tf_savedmodel")
            print(f"Model saved in TensorFlow SavedModel format at {tf_model_path}")
        
        if save_h5_format:
            h5_model_path = os.path.join(tmp_dir, f"{model_name}.h5")
            model.save(h5_model_path, save_format='h5')
            mlflow.log_artifact(h5_model_path)
            print(f"Model saved in H5 format at {h5_model_path}")

        # Clean up saved models after logging them to MLflow
        if save_tf_format and os.path.exists(tf_model_path):
            os.rmdir(tf_model_path)
            print(f"Temporary TensorFlow SavedModel directory {tf_model_path} deleted.")

        if save_h5_format and os.path.exists(h5_model_path):
            os.remove(h5_model_path)
            print(f"Temporary H5 model file {h5_model_path} deleted.")

    def _setup_experiment(self) -> None:
        """
        Private method to set up the experiment by initializing the TensorBoard callback
        and connecting to MLflow (either remote or local).
        """
        # Setup TensorBoard callback
        print(f"TensorBoard callback created. Logs will be saved in {self._tensorboard_logdir}.")

        self.writer = tf.summary.create_file_writer(self._tensorboard_logdir)
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.trace_on(graph=True, profiler=False)

        # Setup MLflow experiment and start a run
        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run()
        print(f"MLflow run started with ID: {self._run.info.run_id}")

    def _check_mlflow_server(self) -> bool:
        """
        Private method to check if the MLflow server is online.
        If the server is not reachable, switch to local logging in the mlruns directory.
        """
        try:
            response = requests.get(self._mlflow_uri)
            if response.status_code == 200:
                print("MLflow server is online.")
            else:
                print(f"MLflow server responded with status code {response.status_code}. Switching to local logging.")
                mlflow.set_tracking_uri('file:' + str(self._mlflow_alt_logdir))  # Switch to local logging
        except requests.ConnectionError:
            print("MLflow server is offline. Switching to local logging.")
            mlflow.set_tracking_uri('file:' + str(self._mlflow_alt_logdir))  # Switch to local logging

    def _close_experiment(self) -> None:
        """
        Private method to end the MLflow run.
        """
        if self._run:
            mlflow.end_run()
            print("MLflow run ended.")
