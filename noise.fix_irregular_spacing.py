import os
import pandas as pd
import subprocess
from subprocess import Popen, PIPE, call


def regulate(f):
    print(f)
    # Execute source command below prior to running script
    # cmd_source='source /opt/minc/1.9.15/minc-toolkit-config.sh'
    # pr= Popen(cmd_source.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # pr.communicate()
    cmd_gunzip='gunzip ' + f
    pr0= Popen(cmd_gunzip.split(' ')) #, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    pr0.communicate()

    for space in ['zspace:spacing','yspace:spacing','xspace:spacing']:
        cmd_del='minc_modify_header -delete {} '.format(space) + f.replace(".gz","")
        prd= Popen(cmd_del.split(' '))
        prd.communicate()
        cmd_insert='minc_modify_header -sinsert {}=regular__ '.format(space) + f.replace(".gz","")
        pri= Popen(cmd_insert.split(' ')) 
        pri.communicate()
        
    cmd_gzip='gzip ' + f.replace(".gz","")
    pr1= Popen(cmd_gzip.split(' ')) #, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    pr1.communicate()



# Execute source command below prior to running script
# cmd_source='source /opt/minc/1.9.15/minc-toolkit-config.sh'

fn = '/home/rpizarro/noise/weights/rap_NN007_ten_epochs/irregular-spacing_mnc-files.txt'

irregular_files = pd.read_csv(fn,header=None)
irregular_files = irregular_files.rename(columns={0:'path'})

for path in irregular_files.path:
    if not os.access(path,os.W_OK):
        print('Haz did not give me permission : {}'.format(path))
        continue
    else:
        print('Making irregular spacing reular for : {}'.format(path))
        regulate(path)

