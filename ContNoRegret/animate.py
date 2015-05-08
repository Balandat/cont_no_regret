'''
Some helper funcitons to produce fance animations

@author: Maximilian Balandat
@date: May 7, 2015
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def save_animation(result, frames=50, interval=20, directory=None, show=True):

    # Creating figure, attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    pltpoints = result.problem.pltpoints
    if pltpoints.shape[1] != 2:
        raise Exception('Can plot densities only for dimension d=2!')
    data = result.problem.data
    
    # Extract some information for the plots
    bbox = result.problem.domain.bbox()
    zmax = np.max(data)   
    # create initial object
    plot = ax.plot_trisurf(pltpoints[:,0], pltpoints[:,1], data[0], cmap=cm.jet)
    # Setting the axes properties
    ax.set_xlim3d(bbox.bounds[0])
    ax.set_xlabel('$s_1$')
    ax.set_ylim3d(bbox.bounds[1])
    ax.set_ylabel('$s_2$')
    ax.set_zlim3d([-0.5, zmax])
    ax.set_zlabel('$x$')
    ax.set_title('pdf animation test')
    
    def update_plot(framenum, data, plot):
        ax.clear()
        plot = ax.plot_trisurf(pltpoints[:,0], pltpoints[:,1], data[framenum], cmap=cm.jet, linewidth=0)
        ax.set_xlim3d(bbox.bounds[0])
        ax.set_xlabel('$s_1$')
        ax.set_ylim3d(bbox.bounds[1])
        ax.set_ylabel('$s_2$')
        ax.set_zlim3d([-0.5, zmax])
        ax.set_zlabel('$x$')
        ax.set_title('pdf animation test')
        return plot
   
   
    # Creating the Animation object
    pdf_ani = animation.FuncAnimation(fig, update_plot, frames, fargs=(data, plot),
                                      interval=interval, blit=False)
    if directory is not None:
        pdf_ani.save('{}animation.mp4'.format(directory), extra_args=['-vcodec', 'libx264'])
    if show:
        plt.show()




