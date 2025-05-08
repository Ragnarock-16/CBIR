import os
from PIL import Image, UnidentifiedImageError
import pandas as pd

#419-15-6(duplicate)+8 = 406 test cbir image

def get_cbir_filename(cbir_gt_file):
    df1 = pd.read_excel(cbir_gt_file,header=None)
    files_names = []
    for val in df1[3]:
        if pd.notnull(val):
            files_names.append(val.lower())

    return files_names


def find_discarded_images(image_dir,excel_dir, log_file="corrupted_images.txt"):
    discard = []
    cbir_file_name = get_cbir_filename(excel_dir)
    print(cbir_file_name)
    with open(log_file, "w") as log:
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif')):
                path = os.path.join(image_dir, fname)
                #check if in CBIR set
                if fname.lower().split(".")[0] in cbir_file_name:
                        print(f"CBIR set: {path}")
                        log.write(path + "\n")
                        discard.append(path)
                else:
                    try:
                        with Image.open(path) as img:
                            img = img.convert('RGB')
                    except :
                        print(f"Corrupted: {path}")
                        log.write(path + "\n")
                        discard.append(path)

    print(f"\nDone. Found {len(discard)} corrupted images.")
    return discard

# === Example usage ===
if __name__ == "__main__":
    image_directory = '/Users/nour/Desktop/HighVision_Corpus_Groundtruth/historicaldataset'
    excel_dir = '/Users/nour/Desktop/HighVision_Corpus_Groundtruth/lipade_images_similaires.xlsx'
    find_discarded_images(image_directory,excel_dir)