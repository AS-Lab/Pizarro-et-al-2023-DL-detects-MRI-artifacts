from pathlib import Path
import os

def get_paths(fn):
    with open(fn) as f:
        files = [os.path.dirname(i.strip().split()[-1]) for i in f]
    return list(set(files))



fn = '/home/rpizarro/noise/sheets/dataset_list29-06-12-19-06.txt'
paths = get_paths(fn)

for p in paths:
    snr_txt_file = os.path.join(p,'aQC/3.0/snr_check_results_PASSED.txt')
    # print(snr_txt_file)
    if not os.path.isfile(snr_txt_file):
        print(p)



