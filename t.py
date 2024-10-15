import tensorflow as tf
from tensorflow.keras.utils import plot_model

from src.model import BottleNeckType, build_res_ae, build_res_disc, build_res_unet

def log_model(models):
    """
    Logs one or more models by printing their summaries and saving images of their architectures.
    The model names are inferred from the model's `name` attribute.

    Args:
    models (Union[tf.keras.Model, List[tf.keras.Model], Tuple[tf.keras.Model, ...]]): The model or list/tuple of models to log.
    """
    # Handle a single model or a list/tuple of models
    if not isinstance(models, (list, tuple)):
        models = [models]  # Make it a list if it's a single model

    for index, model in enumerate(models):
        # Infer model name from the model's `name` attribute or use a default name
        model_name = model.name if model.name and model.name != 'model' else f"model_{index + 1}"
        filename = f'/tmp/{model_name}.png'

        # Print the model summary to console
        print(f"\nLogging Model: {model_name}")
        model.summary()
        
        # Generate and save the model plot
        plot_model(
            model,
            to_file=filename,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
        )
        
        print(f"Model architecture saved to {filename}")

def test():
    o = build_res_ae(bottleneck_type = BottleNeckType.DENSE, initial_padding=10, initial_padding_filters=64)
    log_model(o)
    o = build_res_disc(initial_padding=10, initial_padding_filters=64)
    log_model(o)
    o = build_res_unet(skips=4, initial_padding=10, initial_padding_filters=64)
    log_model(o)
    del o

test()
