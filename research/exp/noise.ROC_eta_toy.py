import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys


save_dir = '/home/rpizarro/noise/prediction/rap_NN008_multiple_artifact/toy_example/'

cols = ['truth','probability','norm_var']

nb_vols = 30

# ground truth
t = np.transpose(np.array([[0]*(nb_vols//2) + [1]*(nb_vols//2)]))

p1 = np.random.normal(0.2, 0.3, nb_vols//2)
p2 = np.random.normal(0.8, 0.3, nb_vols//2)
p = np.concatenate((p1,p2),axis=0)
p[p>1.0] = 1.0
p[p<0.0] = 0.0

s = np.random.normal(0, 0.3, nb_vols)
s = np.absolute(s)
s_min = np.nanmin(s)
s_ptp = np.ptp(s,axis=0)
s = (s - s_min)/(s_ptp)

metrics = np.column_stack((t,p,s)) 

df = pd.DataFrame(data=metrics,columns=cols)
print(df)
fn = os.path.join(save_dir,'prediction_toy.csv')
print('Saving toy example to : {}'.format(fn))
df.to_csv(fn)


fig_nb = int(345)
plt.figure(fig_nb,figsize=(8,6))

tpr = [1.0,1.0,0.9,0.7,0.0]
fpr = [0.7,0.3,0.1,0.0,0.0]
plt.plot(fpr,tpr,'k*-',linewidth=2,markersize=10)

tpr = [1.0,1.0,0.875,0.375,0.0]
fpr = [0.625,0.125,0.0,0.0,0.0]
plt.plot(fpr,tpr,'r*-',linewidth=2,markersize=10)

tpr = [1.0,1.0,1.0,0.667,0.0]
fpr = [0.25,0.25,0.0,0.0,0.0]
plt.plot(fpr,tpr,'g*-',linewidth=2,markersize=10)

plt.axis([-0.1, 1.1, -0.1, 1.1])

plt.tick_params(labelsize=15)
plt.xlabel('false positive rate',fontsize=17)
plt.ylabel('true positive rate',fontsize=17)
plt.legend(['100%, no threshold','75.0%, eta<0.5','35.0%, eta<0.1'],fontsize=15)
plt.tight_layout()
plt.grid(True)

fn = os.path.join(save_dir,'ROC_toy_unc_2.png')
print('Saving to : {}'.format(fn))
plt.savefig(fn)
plt.close(fig_nb)





