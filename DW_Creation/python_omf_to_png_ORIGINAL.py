import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt


def omfdataextraction(filename):
    """
    Original parser from the pre-refactor codebase.
    Kept here for reference; the robust replacement is OMFReader in
    dw_neuron_activation.py.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract numeric header values
    numerical_data = []
    for line in lines:
        if line.startswith('#'):
            parts = line.split()
            for p in parts:
                try:
                    numerical_data.append(float(p))
                except ValueError:
                    pass
        else:
            break

    # Grid dimensions from header
    xnodes = int(numerical_data[0])
    ynodes = int(numerical_data[1])
    znodes = int(numerical_data[2])
    xstep  = numerical_data[3]
    ystep  = numerical_data[4]
    zstep  = numerical_data[5]
    xbase  = numerical_data[6]
    ybase  = numerical_data[7]
    zbase  = numerical_data[8]

    x = np.arange(xnodes) * xstep + xbase
    y = np.arange(ynodes) * ystep + ybase
    z = np.arange(znodes) * zstep + zbase
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Obtain Ms values between "Data Text" markers
    matching       = [s for s in lines if "Data Text" in s]
    Ms_begin_index = lines.index(matching[0]) + 1
    Ms_end_index   = lines.index(matching[1])
    Ms_val         = np.loadtxt(lines[Ms_begin_index:Ms_end_index])
    time_val       = numerical_data[9]

    return x, X, y, Y, z, Z, Ms_val, time_val


# Main loop
N = 0
omf_files_list = []
for file in sorted(os.listdir('.')):
    if fnmatch.fnmatch(file, '*.omf'):
        omf_files_list.append(file)
        N += 1

Sim_time = np.zeros(N)
filename = omf_files_list[-1]
[x, X, y, Y, z, Z, S, t_i] = omfdataextraction(filename)
Total_time = t_i * 1e9

Pulse_signal = []
time_i = []
tp = 1
tw = 0.5
J_val = np.arange(10, 50, 5)

for i in range(N):
    filename = omf_files_list[i]
    print(filename)
    [x, X, y, Y, z, Z, S, t_i] = omfdataextraction(filename)
    save_filename = filename.replace('.omf', '')

    Sim_time[i] = t_i * 1e9
    time_i = np.append(time_i, Sim_time[i])
    n = int(Sim_time[i] / tp)

    print(X.shape)

    X_mat = X[:, :, 0] * 1e9   # convert to nm
    Y_mat = Y[:, :, 0] * 1e9
    print(X_mat.shape)

    mz_1d      = S[:, 2]
    mz_reshaped = np.reshape(mz_1d, (len(z), len(y), len(x)))
    l          = len(z) - 1           # top layer
    mz_2d      = mz_reshaped[l, :, :]

    plt.figure(figsize=(10.24, 1.8))
    plt.pcolor(X_mat, Y_mat, mz_2d, cmap='bwr')
    plt.plot(128 * np.ones(len(y)), y * 1e9, 'k', linewidth=2.5)
    plt.plot(384 * np.ones(len(y)), y * 1e9, 'k', linewidth=2.5)
    plt.title(r"J=" + str(J_val[i]) + r" $\mathrm{MA/cm^2}$")
    plt.xticks([0, 128, 256, 384, 512])
    plt.yticks([0, 64])
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    plt.savefig(save_filename + '.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
