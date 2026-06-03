import cv2
import numpy as np

def process_denoise_hdri(image_path, output_path, light_multiplier=15.0):
    print("Step 1: Loading source panorama...")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    print("Step 2: Running Denoiser (iphone grain reduction)...")
    # fastNlMeansDenoisingColored parameters:
    # h = 3 (Filter strength for luminance), hColor = 3 (Filter strength for color noise)
    # templateWindowSize = 7, searchWindowSize = 21 (AI pixel patch scanning sizes)
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    print("Step 3: Linearizing color curves (Gamma 2.2)...")
    # Convert denoised image to 32-bit float (0.0 to 1.0) and untangle compressed space
    img_32 = img_denoised.astype(np.float32) / 255.0
    img_linear = np.power(img_32, 2.2)
    
    print("Step 4: Isolating physical light emitters...")
    # Calculate pixel luminance using standard color weights
    luminance = 0.2126 * img_linear[:,:,2] + 0.7152 * img_linear[:,:,1] + 0.0722 * img_linear[:,:,0]
    
    # Create a mask targeting the brightest 15% of the room (windows/lights)
    light_mask = np.clip((luminance - 0.85) / 0.15, 0.0, 1.0)
    light_mask = np.expand_dims(light_mask, axis=2) # Match BGR shape
    
    print("Step 5: Adding high-dynamic values into highlights...")
    # Leave room tones perfectly intact, but amplify light sources to cast sharp 3D shadows
    hdri_data = img_linear * (1.0 - light_mask) + (img_linear * light_multiplier * light_mask)
    
    print(f"Step 6: Writing clean 32-bit Radiance HDR file to: {output_path}")
    cv2.imwrite(output_path, hdri_data)
    print("Success! Your HDRI is ready for Blender.")

if __name__ == "__main__":
    # Runs directly on your phone panorama
    process_denoise_hdri("NAME_HERE.jpg", "pano_environment.hdr")