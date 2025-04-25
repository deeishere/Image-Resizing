import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file
def read_image(image_name):
    image = cv2.imread(image_name)
    if image is None:
        print("Error: Image not found.")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate energy map (simple gradient-based)
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
    energy[0, :] = 0
    energy[-1, :] = 0
    energy[:, 0] = 0
    energy[:, -1] = 0

    return energy.astype(np.float32)

def find_greedy_seams(energy, num_seams):
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


def remove_multiple_seams(image, seams):
    rows, cols, _ = image.shape
    num_seams = len(seams)
    new_cols = cols - num_seams
    output = np.zeros((rows, new_cols, 3), dtype=np.uint8)

    # Create a boolean mask of pixels to keep
    mask = np.ones((rows, cols), dtype=bool)

    for seam in seams:
        for row in range(rows):
            col = seam[row]
            # Find the next available pixel to remove if duplicate
            while not mask[row, col]:
                col += 1
                if col >= cols:
                    col = seam[row] - 1
                    while not mask[row, col]:
                        col -= 1
            mask[row, col] = False

    for row in range(rows):
        output[row] = image[row][mask[row]]

    return output


# Function to visualize multiple seams in red on the energy map
def apear_seams(energy, seams):
    seam_visualized = cv2.cvtColor(energy.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for seam in seams:
        for row in range(len(seam)):
            seam_visualized[row, seam[row]] = [255, 0, 0]  
    return seam_visualized

# === Main ===
image_name = 'tower.jpg'  
image = read_image(image_name)

if image is not None:
    num_seams_to_remove = 50

    # Step 1: Energy map
    energy = calculate_energy(image)

    # Step 2: Greedy seam finding
    seams = find_greedy_seams(energy.copy(), num_seams_to_remove)
    energy_with_seams = apear_seams(energy.copy(), seams)

    # Step 3: Remove the seams
    carved_image = remove_multiple_seams(image.copy(), seams)
    energy_after = calculate_energy(carved_image)

    # Step 4: Display results
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(energy, cmap='gray')
    axes[0, 1].set_title("Energy Map")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(energy_with_seams)
    axes[1, 0].set_title("Seams (Red)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(energy_after, cmap='gray')
    axes[1, 1].set_title("Energy After Seam Removal")
    axes[1, 1].axis("off")

    axes[2, 0].imshow(carved_image)
    axes[2, 0].set_title("Final Carved Image")
    axes[2, 0].axis("off")

    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.show()
