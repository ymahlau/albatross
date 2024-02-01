

import os
import shutil



def clean_dir(directory_path):
    # Get the list of all items in the current directory
    items = os.listdir(directory_path)

    for item in items:
        item_path = os.path.join(directory_path, item)
        if item == "fixed_time_models":
            # find larges index
            highest_x = None
            highest_file = None
            # Get the list of files in the specified directory
            files = os.listdir(item_path)

            for file in files:
                # Check if the file matches the pattern "m_x"
                if file.startswith("m_") and file[2:-3].isdigit():
                    current_x = int(file[2:-3])
                    # Update the highest_x and highest_file if a higher x is found
                    if highest_x is None or current_x > highest_x:
                        highest_x = current_x
                        highest_file = os.path.join(item_path, file)
            if highest_file is not None:
                shutil.copyfile(highest_file, os.path.join(directory_path, 'latest.pt'))
            shutil.rmtree(item_path)
        elif item == 'eval_models' or item == 'exit':
            shutil.rmtree(item_path)
        elif os.path.isdir(item_path):
            # recursive call
            clean_dir(item_path)

if __name__ == '__main__':
    parent_dir = 'path'
    clean_dir(parent_dir)
