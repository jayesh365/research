import os
from PIL import Image, ImageDraw, ImageFont

def stitch_images(folder1, folder2, output_folder):
    # Get the list of subfolders
    subfolders1 = sorted([os.path.join(folder1, d) for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
    subfolders2 = sorted([os.path.join(folder2, d) for d in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, d))])

    # Initialize lists to hold images for each index
    indexed_images = [[] for _ in range(5)]

    # Loop through the subfolders and gather images
    for subfolder1, subfolder2 in zip(subfolders1, subfolders2):
        images1 = sorted([os.path.join(subfolder1, f) for f in os.listdir(subfolder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
        images2 = sorted([os.path.join(subfolder2, f) for f in os.listdir(subfolder2) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for i in range(5):
            indexed_images[i].append((Image.open(images1[i]), images1[i], Image.open(images2[i]), images2[i]))

    # Check if there are exactly 5 images in each subfolder
    for images in indexed_images:
        if len(images) != 5:
            print("Error: Each subfolder should contain exactly 5 images.")
            return

    # Create stitched images for each index
    for index, image_pairs in enumerate(indexed_images):
        # Get the size of the first image to calculate the output image size
        img_width, img_height = image_pairs[0][0].size
        text_height = 20  # Height for the text

        # Create a new image with the appropriate size
        stitched_image = Image.new('RGB', (2 * img_width, 5 * (img_height + text_height)))

        # Create a draw object
        draw = ImageDraw.Draw(stitched_image)

        # Place the images in the stitched_image with their paths on top
        for row, (img1, path1, img2, path2) in enumerate(image_pairs):
            y_offset = row * (img_height + text_height)

            # Print the paths of the images
            print(f"Index {index + 1}, Row {row + 1}:")
            print(f"  Path1: {path1}")
            print(f"  Path2: {path2}")

            # Draw text for img1 and img2
            draw.text((0, y_offset), path1, fill="black")
            draw.text((img_width, y_offset), path2, fill="black")

            # Place img1 and img2 side by side
            stitched_image.paste(img1, (0, y_offset + text_height))
            stitched_image.paste(img2, (img_width, y_offset + text_height))

        # Save the stitched image
        output_path = os.path.join(output_folder, f"stitched_image_{index + 1}.jpg")
        stitched_image.save(output_path)
        print(f"Stitched image saved as {output_path}\n")

# Example usage
folder1 = "./outputs/lstm2/"
folder2 = "./outputs/s4d2/"
output_folder = "results"

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

stitch_images(folder1, folder2, output_folder)

