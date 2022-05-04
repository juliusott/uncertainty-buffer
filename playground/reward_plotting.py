import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_recorded_rewards(directory,steps_per_epoch=1000, *keywords):
    valid_recordings = [] 
    num_keys = len([key for key in keywords])           
    for record in os.listdir(directory):
        print(f"num keys {num_keys}")
        i = 0
        for key in keywords:
            if str(key) in record:
                i +=1
                print(key,i)
            else:
                break
        if i == num_keys:
            print(f"add {record}")
            valid_recordings.append(directory+record)
    data = np.concatenate([np.expand_dims(np.load(valid_recording),axis=1) for valid_recording in valid_recordings], axis=1)
    data_extended = np.zeros((data.shape[0]*steps_per_epoch, data.shape[1])) # 1000 = number of steps per epoch
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_extended[i*1000, j] = data[i,j]

    data_extended[data_extended==0] = np.nan
    data_as_frame = pd.DataFrame(data=data_extended).interpolate(method="linear", axis=0)
    data_mean = np.mean(data_as_frame.to_numpy(), axis=1)
    data_std = np.std(data_as_frame.to_numpy(), axis=1)
    max_val = np.amax(data_as_frame.to_numpy())
    min_val = np.amin(data_as_frame.to_numpy())
    return data_mean, data_std, max_val, min_val


env_name = "Humanoid-v3"
sac_mean_uncertainty_noise, sac_std_uncertainty_noise, _, _ = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"uncertainty", "noise", "True")
sac_mean_prioritized_noise, sac_std_prioritized_noise, _, _  = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"prioritized", "noise", "True")
sac_mean_uniform_noise, sac_std_uniform_noise, _, _  = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"uniform", "noise", "True")

sac_mean_uncertainty, sac_std_uncertainty, _, _  = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"uncertainty", "noise", "False")
sac_mean_prioritized, sac_std_prioritized, _, _  = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"prioritized", "noise", "False")
#sac_mean_uniform, sac_std_uniform = get_recorded_rewards("./experiments/", 1000, "SAC", env_name,"uniform", "noise", "False")

td3_mean_uncertainty, td3_std_uncertainty, max_uncertainty, min_uncertainty = get_recorded_rewards("./experiments/", 1000, "TD3", env_name,"uncertainty", "noise", "False")
td3_mean_prioritized, td3_std_prioritized, max_prioritized, min_prioritized  = get_recorded_rewards("./experiments/", 1000, "TD3", env_name,"prioritized", "noise", "False")
td3_mean_uniform, td3_std_uniform, max_uniform, min_uniform  = get_recorded_rewards("./experiments/", 1000, "TD3", env_name,"uniform", "noise", "False")
steps = np.arange(sac_mean_uncertainty.shape[0])
#Plotting
fig,ax = plt.subplots(figsize=(14,9))
ax.set_title(env_name)
ax.set_xlabel("steps")
ax.set_ylabel("reward")

#ax.plot(steps,sac_mean_uncertainty_noise, lw=2,label="SAC MEET noise", color='green')
#ax.fill_between(steps,sac_mean_uncertainty_noise+0.96*sac_std_uncertainty_noise, sac_mean_uncertainty_noise-0.96*sac_std_uncertainty_noise, facecolor='green', alpha=0.5)
#ax.plot(steps,sac_mean_prioritized_noise, lw=2,label="SAC Prioritized noise", color='red')
#ax.fill_between(steps,sac_mean_prioritized_noise+0.96*sac_std_prioritized_noise, sac_mean_prioritized_noise-0.96*sac_std_prioritized_noise, facecolor='red', alpha=0.5)
#ax.plot(steps, sac_mean_uniform_noise, lw=2,label="SAC uniform", color='blue')
#ax.fill_between(steps, sac_mean_uniform_noise+0.96*sac_std_uniform_noise, sac_mean_uniform_noise-0.96*sac_std_uniform_noise, facecolor='blue', alpha=0.5)

ax.plot(steps,td3_mean_uncertainty, lw=2,label="TD3 MEET", color='cyan')
ax.fill_between(steps,np.clip(td3_mean_uncertainty+0.96*td3_std_uncertainty,a_min=min_uncertainty, a_max=max_uncertainty), np.clip(td3_mean_uncertainty-0.96*td3_std_uncertainty, a_min=min_uncertainty, a_max=max_uncertainty), facecolor='cyan', alpha=0.5)
ax.plot(steps,td3_mean_prioritized, lw=2,label="TD3 Prioritized", color='orange')
ax.fill_between(steps,np.clip(td3_mean_prioritized+0.96*td3_std_prioritized,a_min=min_prioritized, a_max=max_prioritized), np.clip(td3_mean_prioritized-0.96*td3_std_prioritized, a_min=min_prioritized, a_max=max_prioritized), facecolor='orange', alpha=0.5)
ax.plot(steps, td3_mean_uniform, lw=2,label="TD3 uniform", color='magenta')
ax.fill_between(steps, np.clip(td3_mean_uniform+0.96*td3_std_uniform,a_min=min_uniform, a_max=min_uniform), np.clip(td3_mean_uniform-0.96*td3_std_uniform,a_min=min_uniform, a_max=max_uniform), facecolor='magenta', alpha=0.5)
plt.legend()
plt.show()