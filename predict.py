import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from heatmap import make_gradcam_heatmap

# Load model
model = load_model("lung_cancer_model_4class.h5")

# Force model build
model(np.zeros((1,224,224,3)))

# IMPORTANT: build model before Grad-CAM
model.predict(np.zeros((1,224,224,3)))
print("Model Layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name}")
print("IO layers")
print(model.inputs)
print(model.outputs)

# IMPORTANT: build model once
dummy = np.zeros((1,224,224,3))
model.predict(dummy)

# Last convolution layer - try different layers for better localization
# Based on model architecture, conv2d_2 is the last conv layer
last_conv_layer_name = "conv2d_2"

# Class names (same order as training)
class_names = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]


def create_lung_mask(image):
    """
    Create a mask to isolate lung region from background
    This helps remove heatmap activations outside the lung
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding for better lung segmentation
    # CT scans: lung tissue is darker, background is darker, body outline is brighter
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours to identify body region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the body region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from largest contour
        body_mask = np.zeros_like(gray)
        cv2.drawContours(body_mask, [largest_contour], -1, 255, -1)
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
        
        # Find internal regions (lung area is inside body)
        # For CT scans, lung regions are darker areas within the body
        lung_mask = np.zeros_like(gray)
        
        # Apply Otsu threshold on the body region only
        masked_region = cv2.bitwise_and(blurred, blurred, mask=body_mask)
        _, lung_thresh = cv2.threshold(masked_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert threshold to get dark regions (lungs)
        lung_thresh = cv2.bitwise_not(lung_thresh)
        
        # Apply body mask to get only internal dark regions
        lung_mask = cv2.bitwise_and(lung_thresh, body_mask)
        
        # Clean up lung mask
        lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)
        lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, kernel)
        
        # If no lungs found, fall back to body mask
        if cv2.countNonZero(lung_mask) < 100:
            lung_mask = body_mask
        
        return lung_mask
    else:
        # Fallback: simple thresholding
        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        return mask


def apply_lung_mask_to_heatmap(heatmap, original_img):
    """
    Apply lung mask to heatmap to remove activations outside lung region
    """
    # Create lung mask from original image
    lung_mask = create_lung_mask(original_img)
    
    # Resize mask to match heatmap dimensions
    lung_mask_resized = cv2.resize(lung_mask, (heatmap.shape[1], heatmap.shape[0]))
    
    # Normalize mask to 0-1
    lung_mask_norm = lung_mask_resized / 255.0
    
    # Apply mask to heatmap (set outside lung regions to 0)
    heatmap_masked = heatmap * lung_mask_norm
    
    # Re-normalize masked heatmap to maintain visual contrast
    if np.max(heatmap_masked) > 0:
        heatmap_masked = heatmap_masked / np.max(heatmap_masked)
    
    return heatmap_masked


def predict(image_path):
    """
    Predict lung cancer from CT scan image with improved heatmap
    """
    print(f"Processing image: {image_path}")
    
    # Preprocess image for model
    img = preprocess_image(image_path)

    # Get predictions
    preds = model.predict(img)[0]
    print(f"Predictions: {preds}")
    
    class_index = np.argmax(preds)
    predicted_class = class_names[class_index]
    confidence = preds[class_index] * 100
    
    print(f"Predicted: {predicted_class} with {confidence:.2f}% confidence")

    # Load original image for heatmap
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    original_img_resized = cv2.resize(original_img, (224, 224))

    # Generate Grad-CAM heatmap
    heatmap_raw = make_gradcam_heatmap(img, model, last_conv_layer_name)
    print(f"Heatmap raw shape: {heatmap_raw.shape}, min: {heatmap_raw.min()}, max: {heatmap_raw.max()}")
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap_raw, (224, 224))
    
    # Apply lung mask to remove activations outside lung region
    heatmap_masked = apply_lung_mask_to_heatmap(heatmap_resized, original_img_resized)
    print(f"Masked heatmap max: {heatmap_masked.max()}")
    
    # Convert to uint8 for colormap
    heatmap_uint8 = np.uint8(255 * heatmap_masked)
    
    # Apply color map
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image with adjustable opacity
    # Use addWeighted for better blending
    superimposed_img = cv2.addWeighted(original_img_resized, 0.6, heatmap_colored, 0.4, 0)
    
    # Save heatmap
    heatmap_path = "static/heatmap.jpg"
    cv2.imwrite(heatmap_path, superimposed_img)
    print(f"Heatmap saved to {heatmap_path}")

    return predicted_class, confidence, "heatmap.jpg"


def test_lung_mask(image_path):
    """
    Test function to visualize the lung mask
    """
    original_img = cv2.imread(image_path)
    if original_img is not None:
        original_img_resized = cv2.resize(original_img, (224, 224))
        
        # Create mask
        mask = create_lung_mask(original_img_resized)
        
        # Save mask
        cv2.imwrite("static/lung_mask_test.jpg", mask)
        print("Lung mask saved to static/lung_mask_test.jpg")
        
        # Show mask overlay on image
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_img_resized, 0.7, mask_3ch, 0.3, 0)
        cv2.imwrite("static/lung_mask_overlay.jpg", overlay)
        print("Mask overlay saved to static/lung_mask_overlay.jpg")
        
        return mask
    return None


# Optional: Print layer names for reference
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Available layers in model:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name}")
    print("="*50)