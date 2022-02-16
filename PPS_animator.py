import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.colors as Colors

plt.ioff()

fig = plt.figure(num=1, figsize = (8,8), dpi = 144)
ax = plt.axes()

plt.tick_params(axis='both',which='both',bottom=False,top=False,right=False,left=False,labelbottom=False,labeltop=False,labelleft=False,labelright=False)


the_data = np.load('PPS_sim.npz')

particle_data = the_data[the_data.files[0]]

NORM = Colors.Normalize(vmin = 0 ,vmax = np.max(particle_data['total_neighbour']))


def anim_func(frame_num,pops):
    pop_data = pops[frame_num]

    ax.cla()
    ax.set_facecolor('black')
    ax.set_xlim(1,1 + 100)
    ax.set_ylim(1,1 + 100)
    ax.scatter(pop_data['x'][:],pop_data['y'][:], c = pop_data['total_neighbour'], s = 3, cmap = cm.Blues_r, norm=NORM)
    ax.grid(False)
    ax.set_title(f'frame_number: {frame_num}',loc='left',fontsize=8,fontweight='bold')

    return ax

Anim = FuncAnimation(fig, anim_func, frames=particle_data.shape[0], fargs=(particle_data,), repeat=False, interval = 50, save_count = particle_data.shape[0])

Anim.save('PPS_anim.mp4')

