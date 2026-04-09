import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap with improved stability and localization
    """
    
    # Get last conv layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        print(f"Using layer: {last_conv_layer.name}")
    except ValueError:
        # Fallback to the last convolutional layer in the model
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                print(f"Using fallback layer: {last_conv_layer.name}")
                break
        else:
            raise ValueError("No convolutional layer found in model")
    
    # Create model that maps input -> conv layer output
    conv_model = tf.keras.models.Model(
        model.inputs,
        last_conv_layer.output
    )
    
    # Get the output of the last conv layer
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        
        # Get final predictions
        preds = model(img_array)
        class_index = tf.argmax(preds[0])
        
        # Loss for the predicted class
        loss = preds[:, class_index]
    
    # Calculate gradients
    grads = tape.gradient(loss, conv_output)
    
    # Handle case where gradients might be None
    if grads is None:
        print("Warning: Gradients are None, using uniform weights")
        # Use equal weights for all channels
        pooled_grads = tf.ones(shape=conv_output.shape[-1]) / conv_output.shape[-1]
    else:
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps
    conv_output = conv_output[0]
    
    # Compute heatmap
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    
    # Apply ReLU to focus on positive contributions
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize heatmap to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    # Add small epsilon to avoid division by zero
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def make_gradcam_heatmap_with_smoothing(img_array, model, last_conv_layer_name, sigma=1.0):
    """
    Generate Grad-CAM heatmap with Gaussian smoothing for cleaner results
    """
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Apply Gaussian blur for smoother heatmap
    import cv2
    heatmap_smoothed = cv2.GaussianBlur(heatmap, (5, 5), sigma)
    
    # Re-normalize
    if np.max(heatmap_smoothed) > 0:
        heatmap_smoothed = heatmap_smoothed / np.max(heatmap_smoothed)
    
    return heatmap_smoothed