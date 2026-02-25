import random
import shutil
from pathlib import Path

src_dir = Path(r"C:\dev\ai\data\CaptchaRaw")
dst_dir = Path(r"C:\dev\ai\data\CaptchaTest")

dst_dir.mkdir(parents=True, exist_ok=True)

all_files = [f for f in src_dir.iterdir() if f.is_file()]
if len(all_files) < 200:
    raise ValueError(f"Not enough files: found {len(all_files)}, need 200")

selected = random.sample(all_files, 200)

for f in selected:
    shutil.move(str(f), dst_dir / f.name)

print(f"Moved {len(selected)} files from {src_dir} to {dst_dir}")
