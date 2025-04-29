import cv2
import numpy as np
import matplotlib.pyplot as plt

# Shared Functions 
def read_image(image_name):
    image = cv2.imread(image_name)
    if image is None:
        print(f"Error: Image {image_name} not found.")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calculate_energy(image):
    gImage = 0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]
    energy = np.zeros_like(gImage, dtype=np.float32)
    rows, cols = gImage.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            a, b, c = gImage[i-1, j-1], gImage[i-1, j], gImage[i-1, j+1]
            d, e, f = gImage[i, j-1], gImage[i, j], gImage[i, j+1]
            g, h, i_pixel = gImage[i+1, j-1], gImage[i+1, j], gImage[i+1, j+1]
            xenergy = a + 2*d + g - c - 2*f - i_pixel
            yenergy = a + 2*b + c - g - 2*h - i_pixel
            energy[i, j] = np.sqrt(xenergy**2 + yenergy**2)
    energy = np.uint8(cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX))
    energy[0, :] = energy[-1, :] = energy[:, 0] = energy[:, -1] = 0
    return energy.astype(np.float32)

def apear_seams(energy, seams):
    seam_visualized = cv2.cvtColor(energy.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for seam in seams:
        for row in range(len(seam)):
            seam_visualized[row, seam[row]] = [255, 0, 0]
    return seam_visualized

# Brute Force Functions 
def find_seams_brute_force(energy, num_seams):
    rows, cols = energy.shape
    seams = []
    for _ in range(num_seams):
        best_seam, min_energy = [], float('inf')
        for start_col in range(cols):
            seam, total_energy = [start_col], energy[0, start_col]
            for row in range(1, rows):
                choices = []
                if seam[-1] > 0:
                    choices.append((energy[row, seam[-1]-1], seam[-1]-1))
                choices.append((energy[row, seam[-1]], seam[-1]))
                if seam[-1] < cols-1:
                    choices.append((energy[row, seam[-1]+1], seam[-1]+1))
                min_energy_pixel = min(choices, key=lambda x: x[0])
                total_energy += min_energy_pixel[0]
                seam.append(min_energy_pixel[1])
            if total_energy < min_energy:
                min_energy, best_seam = total_energy, seam
        seams.append(best_seam)
        energy[np.arange(rows), best_seam] = np.max(energy) + 10000
    return seams

# Dynamic Programming Functions 
def find_seams_dp(energy, num_seams):
    rows, cols = energy.shape
    seams = []
    for _ in range(num_seams):
        M, backtrack = energy.copy(), np.zeros_like(energy, dtype=int)
        for row in range(1, rows):
            for col in range(cols):
                left = M[row-1, col-1] if col > 0 else float('inf')
                up = M[row-1, col]
                right = M[row-1, col+1] if col < cols-1 else float('inf')
                min_energy = min(left, up, right)
                backtrack[row, col] = [left, up, right].index(min_energy) - 1 + col
                M[row, col] += min_energy
        seam = []
        min_col = np.argmin(M[-1])
        seam.append(min_col)
        for row in range(rows-1, 0, -1):
            min_col = backtrack[row, min_col]
            seam.append(min_col)
        seam.reverse()
        seams.append(seam)
        energy[np.arange(rows), seam] = np.max(energy) + 10000
    return seams

# Greedy Functions 
def find_seams_greedy(energy, num_seams):
    rows, cols = energy.shape
    seams = []
    energy_copy = energy.copy()
    for _ in range(num_seams):
        seam = []
        col = np.argmin(energy_copy[0, 1:-1]) + 1
        seam.append(col)
        for row in range(1, rows):
            prev_col = col
            options = []
            if prev_col > 1:
                options.append((energy_copy[row, prev_col-1], prev_col-1))
            options.append((energy_copy[row, prev_col], prev_col))
            if prev_col < cols-2:
                options.append((energy_copy[row, prev_col+1], prev_col+1))
            col = min(options, key=lambda x: x[0])[1]
            seam.append(col)
        seams.append(seam)
        energy_copy[np.arange(rows), seam] = np.max(energy_copy) + 1e5
    return seams

# Remove Seams Functions 
def remove_seams_Brute_DP(image, seams):
    rows, cols, _ = image.shape
    new_cols = cols - len(seams)
    reduced_img = np.zeros((rows, new_cols, 3), dtype=np.uint8)
    for row in range(rows):
        keep_pixels = np.ones(cols, dtype=bool)
        for seam in seams:
            if seam[row] < cols:
                keep_pixels[seam[row]] = False
        if np.sum(keep_pixels) > new_cols:
            keep_pixels[np.where(keep_pixels)[0][-1]] = False  # Remove extra pixel if needed
        elif np.sum(keep_pixels) < new_cols:
            keep_pixels[np.where(~keep_pixels)[0][0]] = True  # Add a pixel back if needed      
        reduced_img[row, :, :] = image[row, keep_pixels, :]
    return reduced_img

def remove_seams_greedy(image, seams):
    for seam in seams:
        mask = np.ones(image.shape[:2], dtype=bool)
        for row in range(image.shape[0]):
            mask[row, seam[row]] = False
        image = image[mask].reshape((image.shape[0], image.shape[1] - 1, image.shape[2]))
    return image

# Menu
image_files = ["castle.jpg", "tower.jpg", "cartoon.jpg", "carving.jpg", "center.jpg",
               "dancers.jpg", "fenster.jpg", "grave.jpg", "museum.jpg", "square.jpg"]

algorithms = {
    1: (find_seams_brute_force, remove_seams_Brute_DP, "Brute-force"),
    2: (find_seams_dp, remove_seams_Brute_DP, "Dynamic Programming"),
    3: (find_seams_greedy, remove_seams_greedy, "Greedy")
}

while True:
    print("\nChoose Image:")
    for idx, name in enumerate(image_files, start=1):
        print(f"{idx}) {name}")
    print("11) Exit")

    img_choice = int(input("Enter image choice: "))
    if img_choice == 11:
        print("Exiting.")
        break
    if not (1 <= img_choice <= 10):
        print("Invalid image choice. Try again.")
        continue

    image_name = image_files[img_choice-1]
    image = read_image(image_name)
    if image is None:
        continue

    while True:
        print("\nChoose Algorithm:")
        for i in range(1, 4):
            print(f"{i}) {algorithms[i][2]}")

        algo_choice = int(input("Enter choice: "))
        if algo_choice not in algorithms:
            print("Invalid algorithm choice. Try again.")
            continue

        num_seams_to_remove = 50
        energy = calculate_energy(image)

        find_seams_func, remove_seams_func, algo_name = algorithms[algo_choice]
        seams = find_seams_func(energy.copy(), num_seams_to_remove) 
        energy_with_seams = apear_seams(energy.copy(), seams)
        image_seam_carved = remove_seams_func(image.copy(), seams)
        energy_after_remove = calculate_energy(image_seam_carved)

        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original - {image_name}", fontsize=11, pad=15)
        axes[0, 1].imshow(energy, cmap='gray')
        axes[0, 1].set_title("Energy Map", fontsize=11, pad=15)
        axes[1, 0].imshow(energy_with_seams)
        axes[1, 0].set_title("Seams Visualized", fontsize=11, pad=15)
        axes[1, 1].imshow(energy_after_remove, cmap='gray')
        axes[1, 1].set_title("Energy After Seam Removal", fontsize=11, pad=15)
        axes[2, 0].imshow(image_seam_carved)
        axes[2, 0].set_title(f"Final Image - {algo_name}", fontsize=11, pad=15)
        axes[2, 1].axis('off')

        plt.suptitle("Seam Carving Results", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.subplots_adjust(hspace=0.4)
        plt.show()
