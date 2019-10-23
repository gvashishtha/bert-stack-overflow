import os
import pandas as pd

to_stack = []
directory_in_str = input('directory with csv files: ')
file_dir = os.fsencode(directory_in_str)

files_skipped_counter = 0
total_files_counter = 0

for _file in os.listdir(file_dir):
    filename = os.fsdecode(_file)
    if filename.endswith(".csv"):
        total_files_counter += 1
        try:
            data = pd.read_csv(os.path.join(directory_in_str, filename),
                               lineterminator='\n', header=0
                               )
            to_stack.append(data)
        except pd.errors.ParserError:
            # data malformed
            files_skipped_counter += 1
            print('skipping file {}'.format(filename))
            continue

print('skipped {} files out of {}'.format(
    files_skipped_counter, total_files_counter))
new_name = 'pandas_concat.csv'
vertical_stack = pd.concat(to_stack, axis=0)

print(vertical_stack.head(5))
vertical_stack.to_csv(os.path.join(directory_in_str, new_name), index=False)
