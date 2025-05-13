import os
from transformers import pipeline
from tqdm import tqdm
import time


def get_image_mask(image_path):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
    # pillow_image = pipe(image_path) # applies mask on input and returns a pillow image
    return pillow_mask



# def save_image(objectId, path, pillow_mask):
#     filename_mask = path + str(objectId) + "mask.png"

#     if not os.path.exists(filename_mask):
#         pillow_mask.save(filename_mask)
#     else:
#         print(f"Skipped: {filename_mask} already exists.")



def save_BRIA_image(obj, path):
    image_path = obj["imageUrl"]
    objectId = obj["objectId"]

    filename_mask = path + str(objectId) + "mask.png"
    if os.path.exists(filename_mask):
            print(f"Skipped: {filename_mask} already exists.")
            return

    pillow_mask = get_image_mask(image_path)
    pillow_mask.save(filename_mask)
    # save_image(objectId, path, pillow_mask)



def run_batched_BRIA_processing(data, save_path, batch_size=10, delay=0):
    """
    Iterates through an array of objects and calls save_BRIA_image in batches.

    Args:
        data (list): List of objects with keys "objectId" and "imageUrl"
        save_path (str): Where to save output images
        batch_size (int): How many to process per batch
        delay (float): Optional delay between batches (in seconds)
    """
    total = len(data)

    for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]

        for obj in batch:
            try:
                save_BRIA_image(obj, save_path)
            except Exception as e:
                print(f"Error processing object {obj.get('objectId')}: {e}")

        if delay:
            time.sleep(delay)