import numpy as np
import os, csv
import nibabel as nib
import sys
import glob
# import scipy.ndimage
import difflib
np.seterr(all='raise')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_files(fn,delimiter):
    with open(fn, 'r') as f:
        x = f.readlines()
    if delimiter == ',':
        x=x[1:]
    x = [(i.split(delimiter)[1].strip(),eval(i.split(delimiter)[0])) for i in x]
    return x


def get_snr(fn,snr_dict):
    if 'original' in fn:
        fn = fn.replace('_original','')
    try:
        snr = snr_dict[fn]
        return fn,snr
    except KeyError:
        print('No SNR computed for : {}'.format(fn))
        return fn,-10.0


def get_snr_list():
    mods = ['t1c','pdw','flr','mtOFF','mtON','t1p','t2w']
    snr_list = []
    for m in mods:
        fn = '/home/rpizarro/noise/sheets/aqc3_snr_mod_{}.csv'.format(m)
        snr_list += get_files(fn,',')
        # print(len(snr_list))
        # print(snr_list[-5:])
    snr = [s[1] for s in snr_list]
    files = [s[0] for s in snr_list]
    snr_dict = dict((k,v) for k,v in zip(files,snr))

    return snr_dict

def save_csv(basename,mylist):
    mylist = [('SNR','clean','intensity','motion ringing','coverage','Path')] + mylist
    save_dir = '/home/rpizarro/noise/sheets/'
    fn = os.path.join(save_dir,basename)
    with open(fn, 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(mylist)


# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

test_fn = '/home/rpizarro/noise/XValidFns/07-08-07-10-20.snr_150_inf/test.art123.txt'
delimiter='***'
test = get_files(test_fn,delimiter)
train_fn = '/home/rpizarro/noise/XValidFns/07-08-07-10-20.snr_150_inf/train.art123.txt'
delimiter='***'
train = get_files(train_fn,delimiter)
valid_fn = '/home/rpizarro/noise/XValidFns/07-08-07-10-20.snr_150_inf/valid.art123.txt'
delimiter='***'
valid = get_files(valid_fn,delimiter)

all_files = test+train+valid
print(all_files[:10])
artifact_list = [s[1] for s in all_files]
files_list_orig = [s[0] for s in all_files]
files_list = []
for s in files_list_orig:
    if 'original' in s:
        s = s.replace('_original','')
    files_list += [s]
    
artifact_dict = dict((k,v) for k,v in zip(files_list,artifact_list))

# print(test)

snr_dict = get_snr_list()

# print(snr_dict.keys())

file_snr_artifact = []

for t in all_files:
    # print(t)
    fn,snr = get_snr(t[0],snr_dict)
    if snr > 0:
        print(fn,snr,artifact_dict[fn])
        file_snr_artifact += [[snr] + artifact_dict[fn] + [fn]]

print(file_snr_artifact[:10])


# 'SNR','clean','intensity','motion ringing','coverage','Path'
clean = [item[0] for item in file_snr_artifact if item[1]]
intensity = [item[0] for item in file_snr_artifact if item[2]]
motion = [item[0] for item in file_snr_artifact if item[3]]
coverage = [item[0] for item in file_snr_artifact if item[4]]

print(len(clean),len(intensity),len(motion),len(coverage))

f = plt.figure()
bins = np.linspace(0,500,100)
plt.hist(motion, bins, alpha=0.3,label='motion')
plt.hist(intensity, bins, alpha=0.7,label='intensity')
plt.hist(coverage, bins, alpha=0.7,label='coverage')
plt.hist(clean, bins, alpha=0.5,label='clean')
plt.legend(loc='upper right')
fn='/home/rpizarro/noise/figs/snr_histogram.07-08-07-10-20.snr_150_inf.pdf'
f.savefig(fn,format='pdf',bbox_inches='tight')

print('Clean SNR : {0:0.1f} +- {1:0.1f} : [{2:0.1f},{3:0.1f}]'.format(np.average(clean),np.std(clean),np.amin(clean),np.amax(clean)))
print('Intensity SNR : {0:0.1f} +- {1:0.1f} : [{2:0.1f},{3:0.1f}]'.format(np.average(intensity),np.std(intensity),np.amin(intensity),np.amax(intensity)))
print('Motion SNR : {0:0.1f} +- {1:0.1f} : [{2:0.1f},{3:0.1f}]'.format(np.average(motion),np.std(motion),np.amin(motion),np.amax(motion)))
print('Coverage SNR : {0:0.1f} +- {1:0.1f} : [{2:0.1f},{3:0.1f}]'.format(np.average(coverage),np.std(coverage),np.amin(coverage),np.amax(coverage)))


save_csv('snr_train_dataset.csv',file_snr_artifact)


