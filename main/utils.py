'''Utils'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.ticker import MaxNLocator

cmap = mpl_colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = mpl_colors.Normalize(vmin=0, vmax=9)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

def plot_sample(in_pic, out_pic, predict=None):
    '''Standard task plotting'''
    def plot_pictures(pictures, labels):
        _, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures), 32))
        for i, (pic, label) in enumerate(zip(pictures, labels)):
            axs[i].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
            axs[i].imshow(np.array(pic), cmap=cmap, norm=norm)
            axs[i].set_title(label)
        plt.show()
    if predict is None:
        plot_pictures([in_pic, out_pic], ['Input', 'Output'])
    else:
        plot_pictures([in_pic, out_pic, predict], ['Input', 'Output', 'Predict'])

def plot_some(pictures, labels=None):
    '''Alternative task plotting'''
    if labels is None:
        labels = [str(i) for i in range(len(pictures))]
    _, axs = plt.subplots(2, int(len(pictures)/2), figsize=(len(pictures), 4.5))
    for ax, pic, label in zip(axs.flat, pictures, labels):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
        ax.imshow(np.array(pic), cmap=cmap, norm=norm)
        ax.set_title(label)
    plt.show()

def set_types(tasks):
    '''Only type 1 is relevant'''
    for task in tasks.values():
        same_shape = all(np.shape(el['input']) == np.shape(el['output']) for el in task['train'])
        constant_shape = len({np.shape(el['output']) for el in task['train']}) == 1
        if same_shape:
            task['type'] = 1
        elif constant_shape:
            task['type'] = 2
        else:
            task['type'] = 3
    print(dict(zip(*np.unique([
        task['type'] for task in tasks.values()], return_counts=True))), '\n')

def process_task(task):
    '''Recoloring effort plus'''
    if 're' in task:
        return
    task['re'] = False
    uni_c = set()

    for not_pair in task['test']:
        ic = np.unique(not_pair['input']).tolist()
        uni_c |= set(ic)

    all_a = []
    all_b = []

    for pair in task['train']:
        input_colors, input_counts = np.unique(pair['input'], return_counts=True)
        output_colors = np.unique(pair['output'])
        a = list(zip(input_colors.tolist(), input_counts))
        a = tuple(x[0] for x in sorted(a, key=lambda x: x[1], reverse=True))
        all_a.append(a)
        all_b.append(tuple(output_colors.tolist()))

    uni_a = set(sum(all_a, ()))
    b_all_1 = np.all([len(x) == 1 for x in all_b])

    intersected = False
    for a, b in zip(all_a, all_b):
        ab = set.intersection(set(a), set(b))
        if len(ab) > 0:
            intersected = True
            break

    if len(task['test']) == 1 and uni_a != uni_c and intersected:
        task['re'] = True
        input_colors, input_counts = np.unique(task['test'][0]['input'], return_counts=True)
        c = list(zip(input_colors.tolist(), input_counts))
        c = tuple(x[0] for x in sorted(c, key=lambda x: x[1], reverse=True))

        colors = []
        commons = set.intersection(*[set(x) for x in all_a + [tuple(uni_c)]])
        for k in c:
            some_present = np.any([k in x for x in all_b])
            all_present = np.all([k in x for x in all_b])
            if (b_all_1 and k != 0) or (
                k not in commons and not all_present) or (
                    some_present and not all_present):
                colors.append(k)

        for i in range(len(task['train'])):
            raw = []
            for k in all_a[i]:
                some_present = np.any([k in x for x in all_b])
                all_present = np.all([k in x for x in all_b])
                if (b_all_1 and k != 0) or (
                    k not in commons and not all_present) or (
                        some_present and not all_present):
                    raw.append(k)

            unused = []
            for digit in range(1, 10):
                if digit not in commons | set(colors) | set(raw):
                    unused.append(digit)

            # Pixel counting
            ipc = task['train'][i]['input']
            opc = task['train'][i]['output']

            for j in range(len(raw)):
                if j < len(colors):
                    ipc = [[colors[j] + 10 if x == raw[j] else x for x in sub] for sub in ipc]
                    opc = [[colors[j] + 10 if x == raw[j] else x for x in sub] for sub in opc]
                else:
                    diff = j - len(colors)
                    if diff >= len(unused):
                        break
                    ipc = [[unused[diff] if (
                        x in colors and x == raw[j]) else x for x in sub] for sub in ipc]
                    opc = [[unused[diff] if (
                        x in colors and x == raw[j]) else x for x in sub] for sub in opc]
            ipc = [[x - 10 if x > 9 else x for x in sub] for sub in ipc]
            opc = [[x - 10 if x > 9 else x for x in sub] for sub in opc]
            task['train'][i]['input'] = ipc
            task['train'][i]['output'] = opc

    new_a = set()
    new_b = set()
    for pair in task['train']:
        ic = np.unique(pair['input']).tolist()
        oc = np.unique(pair['output']).tolist()
        new_a |= set(ic)
        new_b |= set(oc)
    task['all_colors'] = new_a | new_b | uni_c

    task['bgc'] = 0
    if any(0 not in x for x in all_a):
        new_a = []
        for pair in task['train']:
            input_colors, input_counts = np.unique(pair['input'], return_counts=True)
            a = list(zip(input_colors.tolist(), input_counts))
            a = tuple(x[0] for x in sorted(a, key=lambda x: x[1], reverse=True))
            new_a.append(a)
        firsts = [x[0] for x in new_a]
        if len(set(firsts)) == 1:
            task['bgc'] = firsts[0]
