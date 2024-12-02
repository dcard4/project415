import os
import cv2
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path):
    """
    Preprocesses the image: reads in grayscale, resizes (if needed), and ensures proper format.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image

def compare_contours(image1, image2):
    """
    Compares the contours of two images using cv2.matchShapes.
    """
    # Find contours for both images
    contours1, _ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours1) > 0 and len(contours2) > 0:
        contour1 = max(contours1, key=cv2.contourArea)  # Largest contour in image1
        contour2 = max(contours2, key=cv2.contourArea)  # Largest contour in image2
        similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
        return similarity
    return float('inf')  # If no contours found, return large value

def compare_images(image1_path, image2_path):
    """
    Compares two images using SSIM and contour matching.
    """
    # Load and preprocess images
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)

    # Resize images to the same dimensions if necessary
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Compute SSIM
    ssim_value = ssim(image1, image2)

    # Compute contour similarity
    contour_similarity = compare_contours(image1, image2)

    return ssim_value, contour_similarity

def find_most_similar_image(reference_image_path, folder_path):
    """
    Compares the reference image with all images in the folder and identifies the most similar image.
    """
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
            file_path = os.path.join(folder_path, filename)
            try:
                ssim_value, contour_similarity = compare_images(reference_image_path, file_path)
                results.append((filename, ssim_value, contour_similarity))
            except Exception as e:
                print(f"Error comparing {file_path}: {e}")

    # Sort results by SSIM and contour similarity
    results.sort(key=lambda x: (-x[1], x[2]))  # Higher SSIM and lower contour similarity

    return results

# Main Execution
if __name__ == "__main__":
    reference_image_path = "TestD.png"  # Path to the test image
    folder_path = "processed_images"  # Path to the folder containing processed images

    # Compare and find the most similar image
    results = find_most_similar_image(reference_image_path, folder_path)

    print("Similarity Results (Sorted by SSIM and Contour Similarity):")
    for filename, ssim_value, contour_similarity in results:
        print(f"{filename}: SSIM = {ssim_value:.4f}, Contour Similarity = {contour_similarity:.4f}")

    if results:
        most_similar = results[0]
        print(f"\nMost similar image: {most_similar[0]} (SSIM: {most_similar[1]:.4f}, Contour Similarity: {most_similar[2]:.4f})")
