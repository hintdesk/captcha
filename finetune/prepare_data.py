import os
import json
import shutil
from pathlib import Path


def delete_and_create(folder_path):
    """Delete folder entirely and recreate it fresh."""
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"✓ Deleted {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {folder_path}")


def create_metadata_for_folder(folder_path):
    """Create metadata.jsonl for a given folder containing images."""
    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return 0

    data = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".png"):
            text = os.path.splitext(filename)[0]
            data.append({"file_name": filename, "text": text})

    if not data:
        print(f"No PNG files found in {folder_path}")
        return 0

    metadata_path = folder_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Created {len(data)} entries in {metadata_path}")
    return len(data)


def main():
    # Source directory with all images
    source_dir = Path(__file__).parent.parent / "data" / "CaptchaRaw"
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir.resolve()}"
        )

    # Destination directories
    data_dir = Path(__file__).parent.parent / "data" / "CaptchaDatasets"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    # Delete and recreate train and validation folders
    print("Deleting and recreating train and validation folders...\n")
    delete_and_create(train_dir)
    delete_and_create(val_dir)

    # Get all PNG files from source directory
    print("\nCollecting PNG files from source directory...")
    png_files = sorted([f for f in source_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() == ".png"])
    
    if not png_files:
        print(f"No PNG files found in {source_dir}")
        return

    print(f"Found {len(png_files)} PNG files\n")

    # Split files 80/20
    train_split = int(len(png_files) * 0.8)
    train_files = png_files[:train_split]
    val_files = png_files[train_split:]

    # Copy files to train folder
    print(f"Copying {len(train_files)} files to train folder...")
    for file in train_files:
        shutil.copy2(file, train_dir / file.name)
    
    # Copy files to validation folder
    print(f"Copying {len(val_files)} files to validation folder...")
    for file in val_files:
        shutil.copy2(file, val_dir / file.name)

    print("\nCreating metadata.jsonl files...\n")
    
    # Create metadata for train folder
    train_count = create_metadata_for_folder(train_dir)
    
    # Create metadata for validation folder
    val_count = create_metadata_for_folder(val_dir)
    
    print(f"\n--- DATA PREPARATION COMPLETE ---")
    print(f"Train: {train_count} entries ({(train_count/(train_count+val_count)*100):.1f}%)")
    print(f"Validation: {val_count} entries ({(val_count/(train_count+val_count)*100):.1f}%)")
    print(f"Total: {train_count + val_count} entries")


if __name__ == "__main__":
    main()