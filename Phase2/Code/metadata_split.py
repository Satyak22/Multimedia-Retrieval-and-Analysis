import csv
import argparse
import pandas as pd
import os
def walk_directory(image_directory):
    directory = os.fsencode(image_directory)
    full_paths = []
    for image in os.listdir(directory):
        # TODO: Add check for image
        full_paths.append(os.fsdecode(image))
    return full_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    parser.add_argument("csv_file")
    parser.add_argument("result_file")
    args = parser.parse_args()
    if not args.image_folder or not args.csv_file or not args.result_file:
        print("Invalid args/")
        os.Exit(-1)
    image_ids = walk_directory(args.image_folder)
    full_dataset = pd.read_csv(args.csv_file)
    new_df=full_dataset.loc[full_dataset['imageName'].isin(image_ids),:]
    print(full_dataset.shape)
    print(new_df.shape)
    print(len(image_ids))
    new_df.to_csv(args.result_file, index=False)

if __name__ == "__main__":
    main()
