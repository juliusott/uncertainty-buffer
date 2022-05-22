import os
import matplotlib
# Use the pgf backend (must be set before pyplot imported)
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 

nice_fonts = {
    "font.family": "serif",
    "pgf.rcfonts": False,
    "font.size": 8
}
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=True)
y_formatter.set_powerlimits((-5, 3))

def y_fmt(x, y):
    return '{:1.0e}'.format(x).replace('e+04', '0k')

matplotlib.rcParams.update(nice_fonts)
#matplotlib.rcParams.update({"font.size": 12})

plt.style.use('seaborn')

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_recorded_rewards(directory, steps_per_epoch=1000, *keywords):
    #fig, ax = plt.subplots(figsize=(14,7))
    valid_recordings = []
    window_size = 10
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
        data= np.load(valid_recording)
        data = running_mean(data, N=10)
        data = np.expand_dims(data, axis=1)
        print(data.shape)
        if data.shape[0] == steps_per_epoch-window_size+1:
            data_to_concat.append(data)
            #ax.plot(data, label=valid_recording[-7:])
        else:
            print(valid_recording)
    data = np.concatenate(
        data_to_concat,
        axis=1,
    )
    #ax.legend()
    #plt.show()
    data_mean = np.mean(data, axis=1)
    data_std = np.std(data, axis=1)
    max_val = np.amax(data)
    min_val = np.amin(data)
    return data_mean, data_std, max_val, min_val


def plot_data(env_name, alg_name, ax, steps_per_epoch):
    (
        mean_uncertainty,
        std_uncertainty,
        max_uncertainty,
        min_uncertainty,
    ) = get_recorded_rewards(
        "./experiments/", steps_per_epoch, alg_name, env_name, "uncertainty", "noise", "False",
    )
    (
        mean_prioritized,
        std_prioritized,
        max_prioritized,
        min_prioritized,
    ) = get_recorded_rewards(
        "./experiments/", steps_per_epoch, alg_name, env_name, "prioritized", "noise", "False"
    )
    mean_uniform, std_uniform, max_uniform, min_uniform = get_recorded_rewards(
        "./experiments/", steps_per_epoch, alg_name, env_name, "uniform", "noise", "False"
    )
    steps = np.linspace(0,mean_uncertainty.shape[0]/1000, num=mean_uncertainty.shape[0])
    # Plotting
    print(f"size {set_size(600)}")
    ax.set_title(env_name, loc='center', wrap=True, fontsize=8)
    #ax.set_xlabel("million steps")
    #ax.set_ylabel("average return")

    ax.plot(steps, mean_prioritized, lw=2, label="Prioritized", color="red", alpha=0.9)
    ax.fill_between(
        steps,
            mean_prioritized + 0.96 * std_prioritized,
            mean_prioritized - 0.96 * std_prioritized,
        facecolor="red",
        alpha=0.5,
    )
    ax.plot(steps, mean_uniform, lw=2, label="Uniform", color="green", alpha=0.9)
    ax.fill_between(
        steps,
            mean_uniform + 0.96 * std_uniform,

            mean_uniform - 0.96 * std_uniform,
        facecolor="green",
        alpha=0.5,
    )
    ax.plot(steps, mean_uncertainty, lw=2, label="MEET", color="blue", alpha=0.9)
    ax.fill_between(
        steps,
            mean_uncertainty + 0.96 * std_uncertainty,
            mean_uncertainty - 0.96 * std_uncertainty,
        facecolor="blue",
        alpha=0.5,
    )
    if np.amax(mean_uncertainty) > 10000:
        print(f"turn sci mode on {env_name}")
        ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

env_names = ["Humanoid-v3", "Ant-v3", "HalfCheetah-v3", "Walker2d-v3", "InvertedPendulum-v2", "Hopper-v3", "InvertedDoublePendulum-v2", "Swimmer-v3", "HumanoidStandup-v2"]
alg_name = "SAC"
fig, axs = plt.subplots(3, 3, figsize=set_size(400, 1.2))
i,j = 0,0
fig.subplots_adjust(hspace=0.9)
for env_name in env_names:
    if env_name=="Ant-v3":
        steps_per_epoch=2000
    else:
        steps_per_epoch=1000
    plot_data(env_name=env_name, alg_name=alg_name, ax=axs[i][j], steps_per_epoch=steps_per_epoch)
    j+=1
    if j == 3:
        j=0
        i+=1

fig.text(0.5, 0.015, 'million steps', ha='center', va='center', fontsize=10)
fig.text(0.05, 0.5, 'average evaluation reward', ha='center', va='center', rotation='vertical', fontsize=10)
#plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.savefig("Results.pgf", format="pgf", bbox_inches="tight")
plt.show()
"""
env_names = ["Humanoid-v3", "Walker2d-v3", "HalfCheetah-v3"]
alg_name = "SAC"
fig, axs = plt.subplots(1, 3, figsize=(9,3.8))
fig.subplots_adjust(hspace=0.9)
for i, env_name in enumerate(env_names):
    steps_per_epoch=1000
    plot_data(env_name=env_name, alg_name=alg_name, ax=axs[i], steps_per_epoch=steps_per_epoch)

# Set common labels
fig.text(0.5, 0.015, 'million steps', ha='center', va='center')
fig.text(0.06, 0.5, 'average evaluation reward', ha='center', va='center', rotation='vertical')
#plt.tick_params(labelcolor="none", bottom=False, left=False)
# Put a legend to the right of the current axis
# fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("AblationStudy.pdf", format="pdf")
plt.show()
"""