import cv2 as cv
from pathlib import Path
from utils import preprocessing
import pandas as pd
import random

def process_data(input_dir="sig_ver\\signatures", output_dir=".\\sig_processed"):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)

    for writer_folder in input_path.iterdir():
        if writer_folder.is_dir():

            new_writer_folder = out_path / writer_folder.name
            new_writer_folder.mkdir(parents=True, exist_ok=True)
            
            for img_path in writer_folder.iterdir():
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    img = cv.imread(str(img_path))
                    
                    if img is not None:
                        img = preprocessing(img)
                        
                        save_path = new_writer_folder / img_path.name 
                        
                        cv.imwrite(str(save_path), img)
    print("Data processing complete.")

def gen_pairs(data_dir=".\\sig_processed"):
    processed_path = Path(data_dir)
    output_csv = ".\\signature_pairs.csv"

    # Group images by writer dynamically to avoid naming mismatches
    # Sort by the number at the end of the folder name (e.g., 'signatures_1' -> 1)
    writers = sorted(
        [d for d in processed_path.iterdir() if d.is_dir()],
        key=lambda x: int(x.name.split('_')[-1] if '_' in x.name else x.name)
    )

    pairs = []

    for writer_path in writers:
        writer_name = writer_path.name
        
        genuines = sorted(list(writer_path.glob("original_*")))
        forgeries = sorted(list(writer_path.glob("forgeries_*")))

        # --- Generate Positive Pairs (Label 1) ---
        for i in range(len(genuines)):
            for j in range(i + 1, len(genuines)):
                pairs.append({
                    "left_path": str(genuines[i]),
                    "right_path": str(genuines[j]),
                    "label": 1,
                    "writer": writer_name
                })

        # --- Generate Negative Pairs (Label 0) ---
        for gen in genuines:
            for forg in forgeries:
                pairs.append({
                    "left_path": str(gen),
                    "right_path": str(forg),
                    "label": 0,
                    "writer": writer_name
                })

    df = pd.DataFrame(pairs)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Writer-Level Split (Train: 1-45, Test: 46-55)
    # Extract the names of the first 45 folders for training
    train_writers = [w.name for w in writers[:45]]
    test_writers = [w.name for w in writers[45:]]

    train_df = df[df['writer'].isin(train_writers)]
    test_df = df[df['writer'].isin(test_writers)]

    train_df.to_csv("train_pairs.csv", index=False)
    test_df.to_csv("test_pairs.csv", index=False)

    print(f"Total pairs generated: {len(df)}")
    print(f"Training pairs (Writers 1-45): {len(train_df)}")
    print(f"Testing pairs (Writers 46-55): {len(test_df)}")

process_data()
gen_pairs()