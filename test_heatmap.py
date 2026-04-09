import cv2
import numpy as np
from predict import predict, test_lung_mask
import os

def test_heatmap_quality(image_path):
    """
    Test and visualize the heatmap quality
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print("="*50)
    print("Testing Heatmap Quality")
    print("="*50)
    
    # Test lung mask
    print("\n1. Testing lung mask...")
    mask = test_lung_mask(image_path)
    
    # Run prediction
    print("\n2. Running prediction with heatmap...")
    result, confidence, heatmap_path = predict(image_path)
    
    print(f"\n3. Results:")
    print(f"   Prediction: {result}")
    print(f"   Confidence: {confidence:.2f}%")
    
    # Check if heatmap was saved
    if os.path.exists(heatmap_path):
        heatmap_img = cv2.imread(heatmap_path)
        print(f"\n4. Heatmap saved successfully to {heatmap_path}")
        print(f"   Heatmap dimensions: {heatmap_img.shape}")
        
        # Check if heatmap has colored regions
        if np.any(heatmap_img[:,:,2] > 0):  # Red channel (since JET colormap)
            print("   ✓ Heatmap has colored activations")
        else:
            print("   ✗ Heatmap has no colored activations")
    else:
        print(f"\n4. Heatmap not found at {heatmap_path}")
    
    print("\n" + "="*50)
    print("Test complete!")
    print("="*50)

if __name__ == "__main__":
    # Replace with your image path
    test_image = "static/uploaded.jpg"
    test_heatmap_quality(test_image)