import os
import shutil
from pathlib import Path

def find_newest_pt_model(src_folder):
    # Get all files in the source folder
    files = list(Path(src_folder).rglob('*.pt'))

    # Filter out directories, if any
    files = [file for file in files if file.is_file()]

    # Sort files by modification time (newest first)
    newest_model = max(files, key=lambda x: x.stat().st_mtime, default=None)

    return newest_model

def move_model_to_folder(src_model, dest_folder):
    if src_model is not None:
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)

        dest_model = Path(dest_folder) / src_model.name

        # Check if the source file is newer than the destination file
        if not dest_model.exists() or src_model.stat().st_mtime > dest_model.stat().st_mtime:
            # Move the model file to the destination folder
            shutil.move(str(src_model), dest_model)
            print(f"Moved {src_model.name} to {dest_folder}")
        else:
            print(f"{src_model.name} is not newer than the existing file in {dest_folder}. Keeping the original file.")
    else:
        print("No .pt model files found in the source folder.")

if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    source_folder = "./outputs/"
    destination_folder = "./MLOpsProject/models"

    all_models = list(Path(source_folder).rglob('*.pt'))

    # Print the found models for debugging
    print("Found .pt models:")
    for model in all_models:
        print(model)

    # Find the newest .pt model in the source folder
    newest_model = find_newest_pt_model(source_folder)

    # Move the newest model to the destination folder
    move_model_to_folder(newest_model, destination_folder)
