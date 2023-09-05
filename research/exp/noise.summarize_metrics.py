import pandas as pd
import numpy as np
import os

root_dir = '/home/rpizarro/noise/prediction/'

pred_dirs = ['rap_NN007','rap_NN007_prob_model','rap_NN007_prob_rmsprop']

XV_sets = ['test','valid','train']

for p in pred_dirs:
    
    paths_dir = os.path.join(root_dir,p)
    print(paths_dir)

    for XV_set in XV_sets:
    
        print('===We are looking at the {} set==='.format(XV_set))
        set_fn = os.path.join(paths_dir,'label_prob_{}.csv'.format(XV_set))
        df = pd.read_csv(set_fn)

        print('(MSE,CCE) ; ({0:0.3f},{1:0.3f})'.format(np.mean(df['MSE']),np.mean(df['CCE'])))











