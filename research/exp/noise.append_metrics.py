import numpy as np
import pandas as pd
import os


paths_dir = '/home/rpizarro/noise/prediction/rap_NN007/'

XV_sets = ['test','valid','train']
# XV_sets = ['test']

# epsilon for computing categorical cross entropy (CCE)
eps = 10**-10



for XV_set in XV_sets:

    print('===We are looking at the {} set==='.format(XV_set))
    set_fn = os.path.join(paths_dir,'label_prob_mod_{}.csv'.format(XV_set))
    df = pd.read_csv(set_fn)
    col_list = list(df)
    col_list.pop()
    print(col_list)

    cols = ['MSE','CCE','lbl_clean','lbl_intensity','lbl_motion-ringing','lbl_coverage','NN_clean','NN_intensity','NN_motion-ringing','NN_coverage','Path {}'.format(XV_set)]
    df_save = pd.DataFrame(columns=cols)

    for index, row in df.iterrows():
        f = row['Path {}'.format(XV_set)]
        l = row[col_list].values.tolist()[:4]
        out = row[col_list].values.tolist()[4:]
        print(l,out,f)
            
        mse = sum([(i-j)**2 for i,j in zip(l,out)])
        cce = sum([-1*(i*np.log(j+eps)+(1-i)*np.log(1-j+eps)) for i,j in zip(l,out)])

        metric_label_probs_paths = [['{0:0.3f}'.format(mse)] + ['{0:0.3f}'.format(cce)] + ['{0:0.3f}'.format(i) for i in list(l)] + ['{0:0.3f}'.format(i) for i in list(out)] + [f]]
        row_append = pd.DataFrame(metric_label_probs_paths, columns=cols)
        df_save = df_save.append(row_append, ignore_index=True)

        print('==TRUE== clean:[{0:0.3f}]\tintensity:[{1:0.3f}]\tmotion/ringing:[{2:0.3f}]\tcoverage:[{3:0.3f}]'.format(l[0],l[1],l[2],l[3]))
        print('==PRED== clean:[{0:0.3f}]\tintensity:[{1:0.3f}]\tmotion/ringing:[{2:0.3f}]\tcoverage:[{3:0.3f}]'.format(out[0],out[1],out[2],out[3]))
    print(df_save.head())
    df_fn = os.path.join(paths_dir,'label_prob_{}.csv'.format(XV_set))
    df_save.to_csv(df_fn)
    # save_csv(XV_set,label_probs_paths)






