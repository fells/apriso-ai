# Example utility function
import os

def list_files_in_directory(directory_path):
    return [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path)]
