from PIL import Image

def stack_images(path1, path2, spacing=20):
    # Load the images
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    # Get dimensions
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the width of the first image and height = height1 + spacing + height2
    new_image = Image.new('RGB', (width1, height1 + spacing + height2), color=(255, 255, 255))

    # Paste the first image at the top
    new_image.paste(image1, (0, 0))

    # Paste the second image below, with 50px spacing, aligned to the left
    new_image.paste(image2, (0, height1 + spacing))

    return new_image