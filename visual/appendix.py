import numpy as np
import matplotlib.pyplot as plt

fixed_patch = dict(
    severity = 'patch_size',
    patch_size = [0, 5, 15, 25, 35],
    Car = [0.682,0.591,0.504,0.412,0.318],
    Truck = [0.627,0.409,0.205,0.060,0.033],
    Bus = [0.518,0.473,0.516,0.473,0.381],
    Pedestrian = [0.574,0.387,0.252,0.159,0.096],
    Motocycle = [0.489,0.189,0.010,0.001,0.000],
    Bicycle = [0.242,0.101,0.048,0.011,0.012],
    Traffic_Cone = [0.634,0.504,0.433,0.389,0.254],
)

dynamic_patch = dict(
    severity = 'scale',
    scale = [0, 0.1, 0.2, 0.3, 0.4],
    Car = [0.682,0.536,0.395,0.234,0.034],
    Truck = [0.627,0.135,0.000,0.000,0.000],
    Bus = [0.518,0.480,0.236,0.128,0.000],
    Pedestrian = [0.574,0.340,0.218,0.082,0.002],
    Motocycle = [0.489,0.080,0.000,0.000,0.000],
    Bicycle = [0.242,0.018,0.009,0.000,0.000],
    Traffic_Cone = [0.634,0.518,0.423,0.084,0.001],
)


def multi_plot_api(xs, ys, labels, xtitle, ytitle, out_path, legend_size=None, fontsize=None, tick_font_size=None):
    assert isinstance(xs, list or tuple)
    assert isinstance(ys, list or tuple)
    assert isinstance(labels, list or tuple)
    assert len(xs) == len(ys) and len(xs) == len(labels)

    ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    if fontsize is not None:
        ax.set_xlabel(..., fontsize=fontsize)
        ax.set_ylabel(..., fontsize=fontsize)
    for i in range(len(xs)):
        label_ = labels[i].replace('_', '-')
        assert len(xs[i]) == len(ys[i]), f"x doesn't match y in {label_}"
        ax.plot(xs[i], ys[i], label=labels[i], marker='o')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if tick_font_size:
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
    if legend_size:
        plt.legend(prop={'size':legend_size})
    else:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.cla()


def parse_data(results, relative=False, classes=None):
    """
    Args:
        results (dict)
        relative (bool): visualize relative performance drop
    """
    assert isinstance(results, dict)
    # keys = list(results.keys())
    if classes:
        keys = classes
    else:
        keys = list(results.keys())

    xs = []
    ys = []
    labels = []
    for key in keys:
        if key == 'severity' or key == results['severity']:
            continue
        x = results[results['severity']]
        y = results[key]
        if relative:
            y = [y[i] / y[0] for i in range(len(y))]
        xs.append(x)
        ys.append(y)
        labels.append(key)
    
    return xs, ys, labels


if __name__=='__main__':
    
    CLASSES = ['Car', 'Bus', 'Traffic_Cone', 'Pedestrian']
    xs, ys, labels = parse_data(fixed_patch, classes=CLASSES, relative=True) # './visual/appendix/fixed_patch.pdf'
    multi_plot_api(xs, ys, labels, 'patch size', ytitle='AP', out_path='./visual/appendix/fixed_patch.pdf')

    # CLASSES = ['Car', 'Bus', 'Traffic_Cone', 'Pedestrian']
    # xs, ys, labels = parse_data(dynamic_patch, classes=CLASSES, relative=True) # './visual/appendix/dynamic_patch.pdf'
    # multi_plot_api(xs, ys, labels, 'patch scale', ytitle='AP', out_path='./visual/appendix/dynamic_patch.pdf')