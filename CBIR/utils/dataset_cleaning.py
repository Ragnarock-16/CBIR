import os
from PIL import Image, UnidentifiedImageError
import pandas as pd
import argparse

def get_cbir_filename(cbir_gt_file):
    df = pd.read_excel(cbir_gt_file, header=None)
    filenames = []

    for row in df.itertuples(index=False):
        for cell in row:
            if pd.notnull(cell):
                filename = str(cell).strip().lower().split('.')[0]
                filenames.append(filename)
    return filenames


def find_discarded_images(image_dir,excel_dir, log_file="corrupted_images.txt"):
    discard = []
    cbir_file_name = get_cbir_filename(excel_dir)
    with open(log_file, "w") as log:
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif')):
                path = os.path.join(image_dir, fname)
                #check if in CBIR set
                if fname.lower().split(".")[0] in cbir_file_name:
                        print(f"CBIR set: {path}")
                        log.write(path + "\n")
                        discard.append(path)
                
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                except :
                    print(f"Corrupted: {path}")
                    log.write(path + "\n")
                    discard.append(path)

    print(f"\nDone. Found {len(discard)} corrupted images.")
    return discard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, required=True, help="Path to image directory")
    parser.add_argument('-gt', '--ground_truth_path', type=str, required=True, help="Path to GT file")
   
    args = parser.parse_args()

    find_discarded_images(args.image_dir,args.ground_truth_path)