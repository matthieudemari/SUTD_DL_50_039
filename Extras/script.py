import os
import shutil

# Get current working directory
current_dir = os.getcwd()

# Define the destination folder
new_dir = os.path.join(current_dir, 'new')
os.makedirs(new_dir, exist_ok=True)

# Walk through all directories and files starting from current_dir
for root, dirs, files in os.walk(current_dir):
    # Skip the 'Recordings' directories
    if 'Recordings' in dirs:
        dirs.remove('Recordings')  # prevents walking into them

    # Skip the 'new' directory to avoid copying back into itself
    if os.path.commonpath([root, new_dir]) == new_dir:
        continue

    # Compute relative path from current_dir
    rel_path = os.path.relpath(root, current_dir)
    dest_dir = os.path.join(new_dir, rel_path)

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Copy each file
    for file in files:
        src_path = os.path.join(root, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy2(src_path, dest_path)

print(f"Files copied to '{new_dir}' (preserving structure, excluding 'Recordings').")
