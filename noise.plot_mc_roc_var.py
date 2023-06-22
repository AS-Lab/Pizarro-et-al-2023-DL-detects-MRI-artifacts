import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import os,sys
# number of inferences
N = [1,10,20,50,100]

# m is mean and s is standard deviation
AUCm = [0.933,0.937,0.937,0.937,0.938]
AUCs = [0.014,0.002,0.002,0.001,0.001]

SSEm = [0.306,0.035,0.024,0.014,0.010]
SSEs = [0.142,0.005,0.004,0.002,0.002]

NMSEm = [1.350,0.157,0.108,0.062,0.045]
NMSEs = [0.629,0.024,0.017,0.009,0.007]


fig, ax = plt.subplots(figsize=(8,5))
# fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()

color_AUC = 'tab:red'
color_SSE = 'tab:blue'
ax.errorbar(N, AUCm, AUCs,linestyle='-',linewidth=0.5,marker='^',markersize=8,capsize=5,label='AUC',color=color_AUC)
twin1.errorbar(N, SSEm, SSEs, linestyle='-',linewidth=0.5, marker='^',markersize=8,capsize=5,color=color_SSE)

ax.set_xlabel("$N_C$, # of inferences",fontsize=25)
ax.set_ylabel("area under curve ",color=color_AUC,fontsize=25)
twin1.set_ylabel("sum of squared errors",color=color_SSE,fontsize=25)

ax.set_xlim(-5, 105)
ax.set_ylim(0.91, 0.95)
ax.tick_params(axis='x',labelsize=20)
ax.tick_params(axis='y',labelcolor=color_AUC,labelsize=20)
twin1.set_ylim(0.0,0.5)
twin1.tick_params(axis='y',labelcolor=color_SSE,labelsize=20)
ax.grid(True)

# ax.yaxis.label.set_color(p1.get_color())
# twin1.yaxis.label.set_color(p2.get_color())

tkw = dict(size=4, width=1.5)
# ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
# twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
# ax.tick_params(axis='x', **tkw)
# plt.title('$\it{}$'.format(title_str),fontsize=25)
fig.tight_layout()



save_dir = '/data/rpizarro/noise/figs/performance/mc/rap_NN007_single_artifact/clean_percent_050/XV0/valid/nb_classes_02/nb_samples_factor_01.00/split_nb_inferences'
fn = os.path.join(save_dir,'stats_plot.png')
print('Saving to : {}'.format(fn))
plt.savefig(fn)

'''

fig_nb =1001
plt.figure(fig_nb,figsize=(5,8))

ax = plt.subplot(311)
# plt.errorbar(N, AUCm, AUCs, linestyle='None', marker='^',markersize=5,capsize=5)
plt.errorbar(N, AUCm, AUCs,linestyle='-',linewidth=0.5,marker='^',markersize=8,capsize=5)
plt.ylim([0.91,0.95])
plt.ylabel('AUC',fontsize=20)
ax.set_xticklabels([])
plt.tick_params(labelsize=15)
plt.grid(True)
# plt.plot(N,AUCm,'*')

ax = plt.subplot(312)
plt.errorbar(N, SSEm, SSEs, linestyle='-',linewidth=0.5, marker='^',markersize=8,capsize=5)
plt.ylim([0,0.5])
plt.ylabel('SSE',fontsize=20)
ax.set_xticklabels([])
plt.tick_params(labelsize=15)
plt.grid(True)
# plt.plot(N,SSEm,'*')

ax = plt.subplot(313)
plt.errorbar(N, NMSEm, NMSEs, linestyle='-',linewidth=0.5, marker='^',markersize=8,capsize=5)
plt.ylim([0,2.0])
plt.ylabel('NMSE',fontsize=20)
plt.xlabel('N, # MC inferences',fontsize=20)
# plt.plot(N,NMSEm,'*')
plt.tick_params(labelsize=15)
plt.grid(True)
plt.tight_layout()
'''
