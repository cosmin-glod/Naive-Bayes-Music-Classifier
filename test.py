import os
import numpy as np

genres_dir = "./genres_original"

folder_names = []

sorted_folders = sorted(os.listdir(genres_dir))

for genre_folder in sorted_folders:
    genre_folder_path = os.path.join(genres_dir, genre_folder)

    if os.path.isdir(genre_folder_path):
        for file_name in os.listdir(genre_folder_path):
            folder_names.append(genre_folder)

folder_names_array = np.array(folder_names)
# print(folder_names_array, len(folder_names_array))

np.save("labels.npy", folder_names_array)