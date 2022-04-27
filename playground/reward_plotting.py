import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ddpg_1 = np.load("MultiHeadDDPG1.npy")
ddpg_0 = np.load("MultiHeadDDPG.npy")
ddpg_ex_0 = np.zeros((ddpg_0.shape[0]*1000, ))
ddpg_ex_1 = np.zeros((ddpg_1.shape[0]*1000, ))
for i in range(ddpg_0.shape[0]):
    ddpg_ex_0[i*1000] = ddpg_0[i]
    ddpg_ex_1[i*1000] = ddpg_1[i]
ddpg_ex_0[ddpg_ex_0==0] = np.nan
ddpg_ex_1[ddpg_ex_1==0] = np.nan
ddpg_0 = pd.Series(ddpg_ex_0).interpolate(method="cubic")
ddpg_1 = pd.Series(ddpg_ex_1).interpolate(method="cubic")
ddpg = np.concatenate([np.expand_dims(ddpg_0,axis=1), np.expand_dims(ddpg_1,axis=1)], axis=-1)
ddpg_mean = np.mean(ddpg, axis=1)
ddpg_std = np.std(ddpg,axis=1)
ddpg_std[0] = 0
print(ddpg_std[:10])

# Same for SAC
sac_0 = np.load("MultiHeadSAC.npy")
sac_1 = np.load("MultiHeadSAC1.npy")
sac_ex_0 = np.zeros((sac_0.shape[0]*1000, ))
sac_ex_1 = np.zeros((sac_1.shape[0]*1000, ))
for i in range(sac_0.shape[0]):
    sac_ex_0[i*1000] = sac_0[i]
    sac_ex_1[i*1000] = sac_1[i]
sac_ex_0[sac_ex_0==0] = np.nan
sac_ex_1[sac_ex_1==0] = np.nan
sac_0 = pd.Series(sac_ex_0).interpolate(method="cubic")
sac_1 = pd.Series(sac_ex_1).interpolate(method="cubic")
sac = np.concatenate([np.expand_dims(sac_0,axis=1), np.expand_dims(sac_1,axis=1)], axis=-1)
sac_mean = np.mean(sac, axis=1)
sac_std = np.std(sac,axis=1)
sac_std[0] = 0
print(sac_std[:10])


# Same for SAC uniform
sac_u_0 = np.load("playground/SAC.npy")
sac_u_1 = np.load("playground/SAC1.npy")
sac_u_ex_0 = np.zeros((sac_u_0.shape[0]*1000, ))
sac_u_ex_1 = np.zeros((sac_u_1.shape[0]*1000, ))
for i in range(sac_u_0.shape[0]):
    sac_u_ex_0[i*1000] = sac_u_0[i]
    sac_u_ex_1[i*1000] = sac_u_1[i]
sac_u_ex_0[sac_u_ex_0==0] = np.nan
sac_u_ex_1[sac_u_ex_1==0] = np.nan
sac_u_0 = pd.Series(sac_u_ex_0).interpolate(method="cubic")
sac_u_1 = pd.Series(sac_u_ex_1).interpolate(method="cubic")
sac_u = np.concatenate([np.expand_dims(sac_u_0,axis=1), np.expand_dims(sac_u_1,axis=1)], axis=-1)
sac_u_mean = np.mean(sac_u, axis=1)
sac_u_std = np.std(sac_u,axis=1)
sac_u_std[0] = 0
print(sac_std[:10])


#Same for TD3
td3_0 = np.load("MultiHeadTD31std_x_mean.npy")
td3_1 = np.load("MultiHeadTD32std_x_mean.npy")
td3_2 = np.load("MultiHeadTD33std_x_mean.npy")
td3_ex_0 = np.zeros((td3_0.shape[0]*1000, ))
td3_ex_1 = np.zeros((td3_1.shape[0]*1000, ))
td3_ex_2 = np.zeros((td3_2.shape[0]*1000, ))
for i in range(td3_0.shape[0]):
    td3_ex_0[i*1000] = td3_0[i]
    td3_ex_1[i*1000] = td3_1[i]
    td3_ex_2[i*1000] = td3_2[i]
td3_ex_0[td3_ex_0==0] = np.nan
td3_ex_1[td3_ex_1==0] = np.nan
td3_ex_2[td3_ex_2==0] = np.nan
td3_0 = pd.Series(td3_ex_0).interpolate(method="cubic")
td3_1 = pd.Series(td3_ex_1).interpolate(method="cubic")
td3_2 = pd.Series(td3_ex_2).interpolate(method="cubic")
td3 = np.concatenate([np.expand_dims(td3_0,axis=1), np.expand_dims(td3_1,axis=1), np.expand_dims(td3_2,axis=1)], axis=-1)
td3_mean = np.mean(td3, axis=1)
td3_std = np.std(td3,axis=1)
td3_std[0] = 0
print(td3_mean.shape)

#Same for TD3 uniform
td3_u_0 = np.load("MultiHeadTD31std_div_mean.npy")
td3_u_1 = np.load("MultiHeadTD32std_div_mean.npy")
td3_u_2 = np.load("MultiHeadTD33std_div_mean.npy")
td3_u_ex_0 = np.zeros((td3_u_0.shape[0]*1000, ))
td3_u_ex_1 = np.zeros((td3_u_1.shape[0]*1000, ))
td3_u_ex_2 = np.zeros((td3_u_2.shape[0]*1000, ))
for i in range(td3_u_0.shape[0]):
    td3_u_ex_0[i*1000] = td3_u_0[i]
    td3_u_ex_1[i*1000] = td3_u_1[i]
    td3_u_ex_2[i*1000] = td3_u_2[i]
td3_u_ex_0[td3_u_ex_0==0] = np.nan
td3_u_ex_1[td3_u_ex_1==0] = np.nan
td3_u_ex_2[td3_u_ex_2==0] = np.nan
td3_u_0 = pd.Series(td3_u_ex_0).interpolate(method="cubic")
td3_u_1 = pd.Series(td3_u_ex_1).interpolate(method="cubic")
td3_u_2 = pd.Series(td3_u_ex_2).interpolate(method="cubic")
td3_u = np.concatenate([np.expand_dims(td3_u_0,axis=1), np.expand_dims(td3_u_1,axis=1), np.expand_dims(td3_u_2,axis=1)], axis=-1)
td3_u_mean = np.mean(td3_u, axis=1)
td3_u_std = np.std(td3_u,axis=1)
td3_u_std[0] = 0
print(td3_u_std.shape)

td3_uniform = np.load("playground/TD3.npy")
td3_uniform_ex = np.zeros((td3_uniform.shape[0]*1000, ))
for i in range(td3_uniform.shape[0]):
    td3_uniform_ex[i*1000] = td3_uniform[i]
td3_uniform_ex[td3_uniform_ex==0] = np.nan
td3_uniform = pd.Series(td3_uniform_ex).interpolate(method="cubic")

epochs = np.arange(td3_mean.shape[0])
#Plotting
fig,ax = plt.subplots(figsize=(10,7))
#ax.plot(epochs, ddpg_mean,lw=2, label="DDPG", color='blue')
#ax.fill_between(ddpg_mean+ 10*ddpg_std, ddpg_mean-10*ddpg_std, facecolor='blue', alpha=1)
#ax.plot(epochs,sac_mean, lw=1,label="SAC MEET", color='red')
#ax.fill_between(epochs,sac_mean+sac_std, sac_mean-sac_std, facecolor='red', alpha=0.5)
#ax.plot(epochs,sac_u_mean, lw=1,label="SAC uniform", color='orange')
#ax.fill_between(epochs,sac_u_mean+sac_std, sac_u_mean-sac_std, facecolor='orange', alpha=0.5)
ax.plot(epochs,td3_mean, lw=2,label="TD3 mean times variance", color='green')
ax.fill_between(epochs,td3_mean+td3_std, td3_mean-td3_std, facecolor='green', alpha=0.5)
ax.plot(epochs,td3_u_mean, lw=2,label="TD3 variance over mean", color='red')
ax.fill_between(epochs,td3_u_mean+td3_u_std, td3_u_mean-td3_u_std, facecolor='red', alpha=0.5)
ax.plot(epochs,td3_uniform[:200000], lw=2,label="TD3 uniform", color='blue')
#ax.fill_between(td3_mean+10*td3_std, td3_mean-10*td3_std, facecolor='orange', alpha=0.5)
plt.legend()
plt.show()