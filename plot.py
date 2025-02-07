import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import Oranges, Spectral
from pfcorrect import correct


def plot():
    # set up plotting

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    xyticks = np.arange(0, 11, 2)
    zticks = np.arange(0, 1.1, 0.1)

    _p_load = np.linspace(1, 10)
    _q_load = np.copy(_p_load)
    _p_load, _q_load = np.meshgrid(_p_load, _q_load)
    _pf = np.divide(_p_load, np.sqrt(np.add(np.square(_p_load),np.square(_q_load))))


    # plot
    _ = ax1.set(
            title='Before Compensation',
            xlabel='$P_{load}$ (MW)',
            xlim=(0,11),
            xticks=xyticks,
            ylabel='$Q_{load}$ (MVAR)',
            ylim=(0,11),
            yticks=xyticks,
            zlabel='PF',
            zlim=(0, 1.0),
            zticks=zticks
            )
    surf_before = ax1.plot_surface(_p_load, _q_load, _pf, cmap=Spectral, linewidth=0, antialiased=False, rcount=200, ccount=200)

    # adjust default viewing angle
    ax1.view_init(30, 220)
    fig.colorbar(surf_before, location='bottom', ticks=[0, 0.25, 0.5, 0.75, 1.0])

    
    _p_load = np.linspace(1, 10)
    _q_load = np.copy(_p_load)
    _p_load, _q_load = np.meshgrid(_p_load, _q_load)
    _s_load = _p_load + 1j * _q_load
    _q_c = correct(_s_load)
    _q_new = _q_load + _q_c

    _pf = np.divide(_p_load, np.sqrt(np.add(np.square(_p_load), np.square(_q_new))))

    # plot
    _ = ax2.set(
            title='After Compensation',
            xlabel='$P_{load}$ (MW)',
            xticks=xyticks,
            xlim=(0,11),
            ylabel='$Q_{load}$ (MVAR)',
            yticks=xyticks,
            ylim=(0,11),
            zlabel='PF (corrected)',
            zlim=(0, 1.0),
            zticks=zticks
            )
    surf_after = ax2.plot_surface(_p_load, _q_load, _pf, cmap=Spectral, linewidth=0)
    # adjust default viewing angle
    ax2.view_init(30, 220)
    fig.colorbar(surf_after, location='bottom', ticks=[0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()

if __name__ == "__main__":
    plot()
