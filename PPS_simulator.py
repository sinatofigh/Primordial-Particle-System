import numpy as np
import numba as nb

particle_dtype = np.dtype([('frame_id','u2'),('particle_id','u2'),('x','f8'),('y','f8'),('phi','f4'),('R_N','i2'),('L_N','i2'),('total_neighbour','f4'),('delta_phi','f8')])

particles = np.zeros((1100,),dtype=particle_dtype)

biosphere = np.zeros((3000 + 1 , 1100),dtype=particle_dtype)

biosphere[:]['particle_id'] = np.arange(1100)

biosphere[0]['x'] = np.random.uniform(low = 1,high = 100 + 1, size = particles.shape)
biosphere[0]['y'] = np.random.uniform(low = 1,high = 100 + 1, size = particles.shape)
biosphere[0]['phi'] = np.random.uniform(0,360,size=particles.shape)


@nb.njit
def sim_func(universe,tot_frame,pop_size,alfa,beta,velocity,width,height,xmin,ymin,radius):

    for frame_n in range(1,tot_frame + 1):
        universe[frame_n]['frame_id'][:] = frame_n
        np.random.seed(7824 + frame_n)
        rand_idx = np.random.choice(pop_size, size = (pop_size,), replace = False)

        for p in rand_idx:
            for q in rand_idx:
                if universe[frame_n][p]['particle_id'] == universe[frame_n][q]['particle_id']: #flag
                    pass
                else:
                    dx = universe[frame_n - 1][q]['x'] - universe[frame_n - 1][p]['x']
                    if np.abs(dx) > width/2:
                        dx += -np.sign(dx) * width
                    dy = universe[frame_n - 1][q]['y'] - universe[frame_n - 1][p]['y']
                    if np.abs(dy) > height/2:
                        dy += -np.sign(dy) * height
                    ds = np.sqrt(dx**2 + dy**2)
                    if ds <= radius:
                        if (dx*np.sin(np.radians(universe[frame_n - 1][p]['phi'])) - dy*np.cos(np.radians(universe[frame_n - 1][p]['phi']))) < 0:
                            universe[frame_n - 1][p]['R_N'] += 1
                        if (dx*np.sin(np.radians(universe[frame_n - 1][p]['phi'])) - dy*np.cos(np.radians(universe[frame_n - 1][p]['phi']))) > 0:
                            universe[frame_n - 1][p]['L_N'] += 1

            universe[frame_n - 1][p]['total_neighbour'] = universe[frame_n - 1][p]['R_N'] + universe[frame_n - 1][p]['L_N']

            universe[frame_n - 1][p]['delta_phi'] = alfa + (beta * np.sign(universe[frame_n - 1][p]['R_N'] - universe[frame_n - 1][p]['L_N']) * universe[frame_n - 1][p]['total_neighbour'])

            universe[frame_n][p]['phi'] = universe[frame_n - 1][p]['phi'] + universe[frame_n - 1][p]['delta_phi']
            if (universe[frame_n][p]['phi'] >= 360) or (universe[frame_n][p]['phi'] < 0):
                universe[frame_n][p]['phi'] = np.mod(universe[frame_n][p]['phi'],360)
            universe[frame_n][p]['x'] = universe[frame_n - 1][p]['x'] + np.cos(np.radians(universe[frame_n][p]['phi']))
            universe[frame_n][p]['y'] = universe[frame_n - 1][p]['y'] + np.sin(np.radians(universe[frame_n][p]['phi']))
            if ((universe[frame_n][p]['x'] - xmin) > width) or ((universe[frame_n][p]['x'] - xmin) < 0):
                universe[frame_n][p]['x'] += -np.sign(universe[frame_n][p]['x'] - xmin) * width
            if ((universe[frame_n][p]['y'] - ymin) > height) or ((universe[frame_n][p]['y'] - ymin) < 0):
                universe[frame_n][p]['y'] += -np.sign(universe[frame_n][p]['y'] - ymin) * height

    return universe


sim_func(biosphere,3000,1100,180,17,1.0,100,100,1,1,5.0)


results = biosphere[0:3000,:]


np.savez_compressed('PPS_sim.npz', results = results)

