from cut_plot import make_plots

from glob import glob

#process all data sets but the calibration data under this main folder

with open("batch_path.txt", 'r') as path:
    path_text = path.read()

list_dir = glob(path_text+"*/")


for i in reversed(range(len(list_dir))):
    if ("spe" in list_dir[i]) or ("dark" in list_dir[i]): list_dir.pop(i)

print("\n Data to process:\n")
print('\n'.join(list_dir))
print("\n Check the data list above, if not correct, press q then Enter. Otherwise, press any other key to start\n")

flag = input()
if flag == "q": exit()

for data_dir in list_dir:
    print("Now start to process:"+data_dir)

    make_plots(data_dir)