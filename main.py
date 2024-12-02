import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path):
    """
    Preprocesses the image by converting it to grayscale and applying Canny edge detection.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return image, edges

def detect_lines(edges):
    """
    Detects lines in an edge-detected image using Hough Line Transform.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return lines

def draw_lines(image, lines):
    """
    Draws detected lines on the image.
    """
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_image

def extract_features(lines):
    """
    Extracts features (rho and theta of lines) from the detected lines.
    """
    features = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            features.append((rho, theta))
    return np.array(features)

def compare_features(features1, features2):
    """
    Compares two feature sets using Euclidean distance.
    """
    if len(features1) == 0 or len(features2) == 0:
        return float('inf')  # If no features, return a high distance
    return np.linalg.norm(features1 - features2)

def compare_images(image1_path, image2_path):
    """
    Compares two images using SSIM and feature matching.
    """
    # Load and preprocess images
    _, edges1 = preprocess_image(image1_path)
    _, edges2 = preprocess_image(image2_path)

    # Compute SSIM
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    ssim_value = ssim(image1, image2)

    # Detect lines and extract features
    lines1 = detect_lines(edges1)
    lines2 = detect_lines(edges2)
    features1 = extract_features(lines1)
    features2 = extract_features(lines2)

    # Compare features
    feature_distance = compare_features(features1, features2)

    return ssim_value, feature_distance

def process_and_compare(input_image_path, input_folder, output_folder):
    """
    Processes all images in the input folder, compares them to the input image,
    and saves the processed versions to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process and save the input image
    original_image, input_edges = preprocess_image(input_image_path)
    input_lines = detect_lines(input_edges)
    input_lines_image = draw_lines(original_image, input_lines)
    cv2.imwrite(os.path.join(output_folder, "TestD_edges.png"), input_edges)
    cv2.imwrite(os.path.join(output_folder, "TestD_lines.png"), input_lines_image)

    results = []

    # Compare TestD.png to each image in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
            letter_image_path = os.path.join(input_folder, filename)
            letter_name = os.path.splitext(filename)[0]  # Filename without extension

            # Compare TestD with the current letter image
            try:
                ssim_value, feature_distance = compare_images(input_image_path, letter_image_path)
                results.append((letter_name, ssim_value, feature_distance))
                print(f"Compared with {letter_name}: SSIM = {ssim_value:.4f}, Feature Distance = {feature_distance:.4f}")
            except Exception as e:
                print(f"Error comparing with {letter_name}: {e}")

    # Sort results by SSIM (higher is better) and feature distance (lower is better)
    results.sort(key=lambda x: (-x[1], x[2]))

    # Return the most similar letter
    most_similar = results[0] if results else None
    return results, most_similar

# Main Execution
if __name__ == "__main__":
    input_image_path = "TestD.png"        # Path to the test image
    input_folder = "letters"             # Path to the folder containing letter images
    output_folder = "processed_images"   # Path to save processed images

    # Run the process and comparison
    results, most_similar = process_and_compare(input_image_path, input_folder, output_folder)

    # Output results
    print("\nComparison Results:")
    for letter_name, ssim_value, feature_distance in results:
        print(f"{letter_name}: SSIM = {ssim_value:.4f}, Feature Distance = {feature_distance:.4f}")

    if most_similar:
        print(f"\nMost Similar Image: {most_similar[0]} (SSIM = {most_similar[1]:.4f}, Feature Distance = {most_similar[2]:.4f})")
