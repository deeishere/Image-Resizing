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
def find_seams(energy, num_seams):
    rows, cols = energy.shape
    seams = []

    for _ in range(num_seams):  
        best_seam = []
        min_energy = float('inf')

        for start_col in range(cols):
            seam = [start_col]  
            total_energy = energy[0, start_col]

            for row in range(1, rows):
                choices = []
                if seam[-1] > 0:
                    choices.append((energy[row, seam[-1] - 1], seam[-1] - 1))  
                choices.append((energy[row, seam[-1]], seam[-1]))  
                if seam[-1] < cols - 1:
                    choices.append((energy[row, seam[-1] + 1], seam[-1] + 1))  

                min_energy_pixel = min(choices, key=lambda x: x[0])  
                total_energy += min_energy_pixel[0]
                seam.append(min_energy_pixel[1])

            if total_energy < min_energy:
                min_energy = total_energy
                best_seam = seam

        seams.append(best_seam)
        energy[np.arange(rows), best_seam] = np.max(energy) + 10000  

    return seams

# Function to remove multiple seams from the image 
def remove_seams(image, seams):
    rows, cols, _ = image.shape
    new_cols = cols - len(seams)  
    reduced_img = np.zeros((rows, new_cols, 3), dtype=np.uint8)

    for row in range(rows):
        keep_pixels = np.ones(cols, dtype=bool)
        for seam in seams:
            if seam[row] < cols:  # Ensure seam index is within bounds
                keep_pixels[seam[row]] = False  

        # Adjust if the number of pixels is mismatched
        if np.sum(keep_pixels) > new_cols:
            keep_pixels[np.where(keep_pixels)[0][-1]] = False  # Remove extra pixel if needed
        elif np.sum(keep_pixels) < new_cols:
            keep_pixels[np.where(~keep_pixels)[0][0]] = True  # Add a pixel back if needed

        reduced_img[row, :, :] = image[row, keep_pixels, :]

    return reduced_img

# Function to visualize multiple seams in red on the energy map
def apear_seams(energy, seams):
    seam_visualized = cv2.cvtColor(energy.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for seam in seams:
        for row in range(len(seam)):
            seam_visualized[row, seam[row]] = [255, 0, 0]  
    return seam_visualized

# Function to restore color to the resized image
def reverse_grayscale_to_rgb(energy, original_image):
    r_channel = original_image[:, :, 0]
    g_channel = original_image[:, :, 1]
    b_channel = original_image[:, :, 2]

    reversed_r = np.clip(r_channel + energy, 0, 255)
    reversed_g = np.clip(g_channel + energy, 0, 255)
    reversed_b = np.clip(b_channel + energy, 0, 255)

    reversed_image = np.stack([reversed_r, reversed_g, reversed_b], axis=-1)
    return reversed_image.astype(np.uint8)

# Main Section
image_name = 'castle.jpg'
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
    final_image = reverse_grayscale_to_rgb(calculate_energy(image_seam_carved), image_seam_carved)

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
    axes[2, 0].imshow(final_image)
    axes[2, 0].set_title("5) Final Image (Width Reduced, Colors Restored)")
    axes[2, 0].axis('off')

    # Hide extra empty subplot
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()
