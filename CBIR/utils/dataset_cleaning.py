import os
from PIL import Image, UnidentifiedImageError

def find_corrupted_images(image_dir, log_file="corrupted_images.txt"):
    corrupted = []

    with open(log_file, "w") as log:
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif')):
                path = os.path.join(image_dir, fname)
                try:
                    with Image.open(path) as img:
                         img = img.convert('RGB')
                except :
                    print(f"Corrupted: {path}")
                    log.write(path + "\n")
                    corrupted.append(path)

    print(f"\nDone. Found {len(corrupted)} corrupted images.")
    return corrupted

# === Example usage ===
if __name__ == "__main__":
    try: 
        with Image.open('/Volumes/U1/Fac/M2/HighVision_Corpus_Groundtruth/historicaldataset/pho_2K47161_59_02.jpg') as img:
            img = img.convert('RGB')
    except:
        print('fuck')
    image_directory = '/Volumes/U1/Fac/M2/HighVision_Corpus_Groundtruth/historicaldataset'
    find_corrupted_images(image_directory)