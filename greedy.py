
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read an image
def read_image(image_name):
    image = cv2.imread(image_name)
    if image is None:
        print("Error: Image not found.")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to convert to grayscale and calculate energy map
def calculate_energy(image):
    gImage = 0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]
    energy = np.zeros_like(gImage, dtype=np.float32)

    rows, cols = gImage.shape  
    for i in range(1, rows - 1):  
        for j in range(1, cols - 1):
            a = gImage[i - 1, j - 1]  
            b = gImage[i - 1, j]      
            c = gImage[i - 1, j + 1]  
            d = gImage[i, j - 1]      
            e = gImage[i, j]          
            f = gImage[i, j + 1]      
            g = gImage[i + 1, j - 1]  
            h = gImage[i + 1, j]      
            i_pixel = gImage[i + 1, j + 1]  

            xenergy = a + 2 * d + g - c - 2 * f - i_pixel
            yenergy = a + 2 * b + c - g - 2 * h - i_pixel

            energy[i, j] = np.sqrt(xenergy**2 + yenergy**2)

    energy = np.uint8(cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX))

    # Set edges to 0
    energy[0, :] = 0    
    energy[-1, :] = 0   
    energy[:, 0] = 0    
    energy[:, -1] = 0   

    return energy.astype(np.float32)

# Function to find multiple seams with the lowest energy (brute force)
import numpy as np

def find_seams(energy, num_seams):
    rows, cols = energy.shape
    seams = []
    energy_copy = energy.copy()

    for _ in range(num_seams):
        seam = []

        # Start from 1 to cols-2 (avoid edge start)
        col = np.argmin(energy_copy[0, 1:-1]) + 1
        seam.append(col)

        for row in range(1, rows):
            prev_col = col
            options = []

            # consider neighbors within 1 and cols-2
            if prev_col > 1:
                options.append((energy_copy[row, prev_col - 1], prev_col - 1))
            options.append((energy_copy[row, prev_col], prev_col))
            if prev_col < cols - 2:
                options.append((energy_copy[row, prev_col + 1], prev_col + 1))

            col = min(options, key=lambda x: x[0])[1]
            seam.append(col)

        seams.append(seam)

        # Avoid reusing same seam
        energy_copy[np.arange(rows), seam] = np.max(energy_copy) + 1e5

    return seams



# Function to remove multiple seams from the image 
def remove_seams(image, seams):
    for seam in seams:
        mask = np.ones(image.shape[:2], dtype=bool)
        for row in range(image.shape[0]):
            mask[row, seam[row]] = False
        image = image[mask].reshape((image.shape[0], image.shape[1] - 1, image.shape[2]))
    return image


# Function to visualize multiple seams in red on the energy map
def apear_seams(energy, seams):
    seam_visualized = cv2.cvtColor(energy.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for seam in seams:
        for row in range(len(seam)):
            seam_visualized[row, seam[row]] = [255, 0, 0]  
    return seam_visualized




# Main Section
image_name = 'dancers.jpg'
image = read_image(image_name)

if image is not None:
    num_seams_to_remove = 50  

    # Step 1: Compute initial energy map
    energy = calculate_energy(image)

    # Step 2: Find and visualize multiple seams
    seams = find_seams(energy.copy(), num_seams_to_remove)
    energy_with_seams = apear_seams(energy.copy(), seams)

    # Step 3: Remove seams
    image_seam_carved = remove_seams(image.copy(), seams)

    # Step 4: Restore original colors to resized image
    

    # Compute energy map after seam removal
    energy_after_remove = calculate_energy(image_seam_carved)

    # Display the steps
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Step 1: Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("1) Start with an image")
    axes[0, 0].axis('off')

    # Step 2: Energy Map
    axes[0, 1].imshow(energy, cmap='gray')
    axes[0, 1].set_title("2) Compute the energy value for each pixel")
    axes[0, 1].axis('off')

    # Step 3: Visualization of multiple red seams on grayscale
    axes[1, 0].imshow(energy_with_seams)
    axes[1, 0].set_title("3) show seams")
    axes[1, 0].axis('off')

    # Step 4: Energy Map After Seam Removal
    axes[1, 1].imshow(energy_after_remove, cmap='gray')
    axes[1, 1].set_title("4) Remove seams")
    axes[1, 1].axis('off')

    # Step 5: Final Seam-Carved Image with Restored Colors
    axes[2, 0].imshow(image_seam_carved)
    axes[2, 0].set_title("5) Final Image")
    axes[2, 0].axis('off')

    # Hide extra empty subplot
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()
