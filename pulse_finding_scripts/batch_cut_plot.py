from cut_plot import make_plots

from glob import glob

#process all data sets but the calibration data under this main folder

with open("batch_path.txt", 'r') as path:
    path_text = path.read()
list_dir = glob(path_text)
for dir in list_dir:
    if ("spe" in dir) or ("dark" in dir): list_dir.remove(dir)
print('Data to process:', '\n'.join(list_dir))

for data_dir in list_dir:
    print("Now start to process:"+data_dir)

    make_plots(data_dir)