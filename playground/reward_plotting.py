import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_recorded_rewards(directory, steps_per_epoch=1000, *keywords):
    valid_recordings = []
    num_keys = len([key for key in keywords])
    for record in os.listdir(directory):
        i = 0
        for key in keywords:
            if str(key) in record:
                i += 1
            else:
                break
        if i == num_keys:
            print(f"add {record}")
            valid_recordings.append(directory + record)

    data_to_concat = []
    for valid_recording in valid_recordings:
        data = np.expand_dims(np.load(valid_recording), axis=1)
        if data.shape[0] == 1000:
            data_to_concat.append(data)
        else:
            print(valid_recording)
    data = np.concatenate(
        data_to_concat,
        axis=1,
    )
    data_extended = np.zeros(
        (data.shape[0] * steps_per_epoch, data.shape[1])
    )  # 1000 = number of steps per epoch
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_extended[i * 1000, j] = data[i, j]

    data_extended[data_extended == 0] = np.nan
    data_as_frame = pd.DataFrame(data=data_extended).interpolate(
        method="linear", axis=0
    )
    data_mean = np.mean(data_as_frame.to_numpy(), axis=1)
    data_std = np.std(data_as_frame.to_numpy(), axis=1)
    max_val = np.amax(data_as_frame.to_numpy())
    min_val = np.amin(data_as_frame.to_numpy())
    return data_mean, data_std, max_val, min_val


env_name = "Walker2d-v3"
alg_name = "SAC"

(
    mean_uncertainty,
    std_uncertainty,
    max_uncertainty,
    min_uncertainty,
) = get_recorded_rewards(
    "./experiments/", 1000, alg_name, env_name, "uncertainty", "noise", "False"
)
(
    mean_prioritized,
    std_prioritized,
    max_prioritized,
    min_prioritized,
) = get_recorded_rewards(
    "./experiments/", 1000, alg_name, env_name, "prioritized", "noise", "False"
)
mean_uniform, std_uniform, max_uniform, min_uniform = get_recorded_rewards(
    "./experiments/", 1000, alg_name, env_name, "uniform", "noise", "False"
)
steps = np.arange(mean_uncertainty.shape[0])
# Plotting
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_title(env_name)
ax.set_xlabel("steps")
ax.set_ylabel("cumulative reward")

# ax.plot(steps,sac_mean_uncertainty_noise, lw=2,label="SAC MEET noise", color='green')
# ax.fill_between(steps,sac_mean_uncertainty_noise+0.96*sac_std_uncertainty_noise, sac_mean_uncertainty_noise-0.96*sac_std_uncertainty_noise, facecolor='green', alpha=0.5)
# ax.plot(steps,sac_mean_prioritized_noise, lw=2,label="SAC Prioritized noise", color='red')
# ax.fill_between(steps,sac_mean_prioritized_noise+0.96*sac_std_prioritized_noise, sac_mean_prioritized_noise-0.96*sac_std_prioritized_noise, facecolor='red', alpha=0.5)
# ax.plot(steps, sac_mean_uniform_noise, lw=2,label="SAC uniform", color='blue')
# ax.fill_between(steps, sac_mean_uniform_noise+0.96*sac_std_uniform_noise, sac_mean_uniform_noise-0.96*sac_std_uniform_noise, facecolor='blue', alpha=0.5)

ax.plot(steps, mean_uncertainty, lw=2, label=f"{alg_name} MEET", color="blue")
ax.fill_between(
    steps,
        mean_uncertainty + 0.96 * std_uncertainty,
        mean_uncertainty - 0.96 * std_uncertainty,
    facecolor="blue",
    alpha=0.5,
)
ax.plot(steps, mean_prioritized, lw=2, label=f"{alg_name} Prioritized", color="red")
ax.fill_between(
    steps,
        mean_prioritized + 0.96 * std_prioritized,
        mean_prioritized - 0.96 * std_prioritized,
    facecolor="red",
    alpha=0.5,
)
ax.plot(steps, mean_uniform, lw=2, label=f"{alg_name} uniform", color="green")
ax.fill_between(
    steps,
        mean_uniform + 0.96 * std_uniform,

        mean_uniform - 0.96 * std_uniform,
    facecolor="green",
    alpha=0.5,
)
plt.legend()
plt.show()
