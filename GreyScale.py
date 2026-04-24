from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your image file
image_path = 'MonaLisa.jpg'

try:
    # Open the image using Pillow
    color_image_file = Image.open(image_path)

    # Convert the image to a NumPy array
    color_image_array = np.array(color_image_file)

    # Ensure the image is in RGB format (handle cases where it might be RGBA or grayscale)
    if color_image_array.ndim == 2: # Already grayscale
        print("Image is already grayscale. JFIF greyscaling is not applicable.")
        plt.figure(figsize=(5, 5))
        plt.imshow(color_image_array, cmap='gray')
        plt.title('Original Grayscale Image')
        plt.axis('off')
        plt.show()
    elif color_image_array.shape[-1] == 4: # RGBA, convert to RGB
        color_image_array = color_image_array[:, :, :3]
    elif color_image_array.shape[-1] != 3: # Handle other unexpected formats
        print(f"Warning: Image has {color_image_array.shape[-1]} channels. Attempting to convert to RGB.")
        color_image_file = color_image_file.convert('RGB')
        color_image_array = np.array(color_image_file)

    # Display the original color image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(color_image_array)
    plt.title(f'Original Color Image: {image_path}')
    plt.axis('off')

    # Apply the JFIF greyscaling formula
    # Y' = 0.299*R' + 0.587*G' + 0.114*B'
    grayscale_image_jfif = (0.299 * color_image_array[:, :, 0] + \
                            0.587 * color_image_array[:, :, 1] + \
                            0.114 * color_image_array[:, :, 2]).astype(np.uint8)

    # Display the greyscaled image
    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image_jfif, cmap='gray')
    plt.title('Greyscaled (Y\') Image (JFIF)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found. Please upload the image or provide the correct path.")
except Exception as e:
    print(f"An error occurred: {e}")
