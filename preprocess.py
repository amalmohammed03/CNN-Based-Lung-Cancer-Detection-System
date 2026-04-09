import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image for model input
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Invalid image path: {image_path}")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def preprocess_with_enhancement(image_path):
    """
    Enhanced preprocessing with contrast enhancement for CT scans
    """
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Invalid image path: {image_path}")
    
    # Convert to grayscale for enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3-channel
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Resize
    enhanced_3ch = cv2.resize(enhanced_3ch, (224, 224))
    
    # Normalize
    enhanced_3ch = enhanced_3ch / 255.0
    
    # Add batch dimension
    enhanced_3ch = np.expand_dims(enhanced_3ch, axis=0)
    
    return enhanced_3ch