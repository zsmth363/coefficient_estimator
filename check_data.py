import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd

def find_excels_with_nans(folder_path):
    """
    Scan all Excel files in a folder and return a list of filenames
    that contain any NaN values.
    """
    files_with_nans = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.xls', '.xlsx')):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path)
                if (df[df.columns[2]] == 0).all():
                    print(filename)
                # plt.plot(df[df.columns[2]])
                # if df[df.columns[2]].mean() < 0.3:
                #     print(filename)
                # if df.isna().any().any():  # check if any NaN exists
                #     files_with_nans.append(filename)
            except Exception as e:
                print(f"⚠️ Could not read {filename}: {e}")
    
    return files_with_nans



def copy_nonzero_excel_files(src_folder, dest_folder="data"):
    """
    Checks each Excel file in src_folder and copies it to dest_folder
    if the third column does NOT consist entirely of zeros.
    """
    # Create destination folder if it doesn’t exist
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate through Excel files
    for filename in os.listdir(src_folder):
        if filename.endswith((".xls", ".xlsx")):
            file_path = os.path.join(src_folder, filename)

            try:
                # Read the Excel file
                df = pd.read_excel(file_path)

                # Ensure the file has at least 3 columns
                if df.shape[1] < 3:
                    print(f"Skipping {filename}: less than 3 columns")
                    continue

                # Select the third column
                third_col = df.iloc[:, 2]

                # Check if all values are zero
                if not (third_col == 0).all():
                    # Copy to destination folder
                    shutil.copy(file_path, os.path.join(dest_folder, filename))
                    print(f"Copied {filename} → {dest_folder}")
                else:
                    print(f"Skipped {filename}: third column all zeros")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
folder = r"""C:\Users\Zach\OneDrive - Clemson University\IM_runs\IM_results_10-6"""
path = r"""C:\Users\Zach\OneDrive - Clemson University\IM_runs"""
copy_nonzero_excel_files(folder,path)
# bad_files = find_excels_with_nans(folder)
# print("Files with NaN values:", bad_files)