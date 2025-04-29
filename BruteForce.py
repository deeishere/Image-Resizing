import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read an image 
def read_image(image_name):
    image = cv2.imread(image_name)  
    if image is None:
        print("Error: Image not found.")
    else:
        RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        return RGBimage

# Function to convert to grayscale and calculate energy map 
def calulate_energy(image):
   
    #gImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # Convert to grayscale
    gImage =  0.2989 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,0]
    
    energy = np.zeros_like(gImage, dtype=np.float32) # Initialize an empty energy map

    
    rows, cols = gImage.shape  #  energy formula for each pixel 

    for i in range(1, rows - 1):  # Skip first and last row
        for j in range(1, cols - 1):  # Skip first and last column
            
            a = gImage[i - 1, j - 1]  # top-left
            b = gImage[i - 1, j]      # top
            c = gImage[i - 1, j + 1]  # top-right
            d = gImage[i, j - 1]      # left
            e = gImage[i, j]          # center (E)
            f = gImage[i, j + 1]      # right
            g = gImage[i + 1, j - 1]  # bottom-left
            h = gImage[i + 1, j]      # bottom
            i_pixel = gImage[i + 1, j + 1]  # bottom-right

            # Calculate the energy in the x and y directions using the given formula
            xenergy = a + 2 * d + g - c - 2 * f - i_pixel
            yenergy = a + 2 * b + c - g - 2 * h - i_pixel

            # The total energy 
            energy[i, j] = abs(xenergy) + abs(yenergy)

    
    energy = np.uint8(cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX))


    # Set the edges of the energy map to black
    energy[0, :] = 0  # Top edge
    energy[-1, :] = 0  # Bottom edge
    energy[:, 0] = 0  # Left edge
    energy[:, -1] = 0  # Right edge

    return energy


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read an image
def read_image(image_name):
    image = cv2.imread(image_name)
    if image is None:
        print("Error: Image not found.")
    else:
        RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return RGBimage

# Function to convert to grayscale and calculate energy map
def calulate_energy(image):
    # Convert to grayscale using the standard RGB to grayscale conversion
    gImage = 0.2989 * image[:,:,2] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,0]
    
    energy = np.zeros_like(gImage, dtype=np.float32)  # Initialize an empty energy map

    rows, cols = gImage.shape  # Energy formula for each pixel
    for i in range(1, rows - 1):  # Skip first and last row
        for j in range(1, cols - 1):  # Skip first and last column
            a = gImage[i - 1, j - 1]  # top-left
            b = gImage[i - 1, j]      # top
            c = gImage[i - 1, j + 1]  # top-right
            d = gImage[i, j - 1]      # left
            e = gImage[i, j]          # center (E)
            f = gImage[i, j + 1]      # right
            g = gImage[i + 1, j - 1]  # bottom-left
            h = gImage[i + 1, j]      # bottom
            i_pixel = gImage[i + 1, j + 1]  # bottom-right

            # Calculate the energy in the x and y directions using the given formula
            xenergy = a + 2 * d + g - c - 2 * f - i_pixel
            yenergy = a + 2 * b + c - g - 2 * h - i_pixel

            # The total energy 
            energy[i, j] = abs(xenergy) + abs(yenergy)

    energy = np.uint8(cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX))

    # Set the edges of the energy map to black
    energy[0, :] = 0  # Top edge
    energy[-1, :] = 0  # Bottom edge
    energy[:, 0] = 0  # Left edge
    energy[:, -1] = 0  # Right edge

    return energy

# Function to reverse grayscale energy to the original color (attempting to reconstruct)
def reverse_grayscale_to_rgb(energy, original_image):
   # Assuming energy relates to the grayscale intensity, we reverse the process:



    # Step 1: Smooth the energy map using a Gaussian filter to simulate "reversing" edge-detection
    smoothed_energy = cv2.GaussianBlur(energy, (5, 5), 0)  # Smooth the energy map

    # Step 2: Normalize the energy to make sure it's in a usable range for image processing
    smoothed_energy = np.uint8(cv2.normalize(smoothed_energy, None, 0, 255, cv2.NORM_MINMAX))

    # Step 3: Get the individual channels from the original image
    r_channel = original_image[:,:,0]
    g_channel = original_image[:,:,1]
    b_channel = original_image[:,:,2]

    # Step 4: Use the smoothed energy map to enhance or modify the color channels
    # We apply a multiplication factor based on the smoothed energy instead of adding directly
    energy_factor_r = np.clip(1 + smoothed_energy / 255.0, 0, 2)  # Apply scaling based on energy map
    energy_factor_g = np.clip(1 + smoothed_energy / 255.0, 0, 2)
    energy_factor_b = np.clip(1 + smoothed_energy / 255.0, 0, 2)

    # Step 5: Modify the original color channels based on the energy factors
    # Multiply the channels by the energy factors (This enhances edges)
    enhanced_r = np.clip(r_channel * energy_factor_r, 0, 255)
    enhanced_g = np.clip(g_channel * energy_factor_g, 0, 255)
    enhanced_b = np.clip(b_channel * energy_factor_b, 0, 255)

    # Step 6: Combine the adjusted channels back into an RGB image
    enhanced_image = np.stack([enhanced_r, enhanced_g, enhanced_b], axis=-1)

    return enhanced_image.astype(np.uint8)

def reverse_grayscale_to_rgb(energy, original_image):
    # to get the RGB color individual 
    r_channel = original_image[:, :, 0]
    g_channel = original_image[:, :, 1]
    b_channel = original_image[:, :, 2]
    
    #to restore color
    reversed_r = np.clip(r_channel + energy, 0, 255)
    reversed_g = np.clip(g_channel + energy, 0, 255)
    reversed_b = np.clip(b_channel + energy, 0, 255)



    reversed_image = np.stack([reversed_r, reversed_g, reversed_b], axis=-1)
    return reversed_image.astype(np.uint8)

# Main Section
image_name = 'tower.jpg'  
image = read_image(image_name)

if image is not None:
 
    energy = calulate_energy(image)
    ori= reverse_grayscale_to_rgb(energy,image)

    plt.subplot(2, 2, 1)
    plt.imshow(ori)  # Display original image
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(energy, cmap='gray')  # Display energy map
    plt.title('Energy Map (Edges Black)')
    plt.axis('off')



    plt.show()

 
 
