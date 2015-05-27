'''
Some helper functions to produce fancy animations

@author: Maximilian Balandat
@date: May 8, 2015
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from ContNoRegret.Domains import nBox, UnionOfDisjointnBoxes

def save_animations(results, length=10, directory=None, show=False, **kwargs):
    """ Takes in a list of Result objects and creates and saves an animation of the 
        evolution of the pdf over time of total duration length seconds. """
    # save all result animations (for now)
    for r, result in enumerate(results):
        T = result.problem.T
        frames = T-1
        interval = length/T*1000
        try:
            pltdata = result.pltdata             
            pltpoints = result.problem.pltpoints
            
            # Creating figure, attaching 3D axis to the figure
            fig = plt.figure()
            ax = p3.Axes3D(fig)        

            # Extract some information for the plots
            bbox = result.problem.domain.bbox()
            # idk why the FUCK this does not work just using np arrays!?
            zmax = np.max([np.max([np.max(df) for df in dflat]) for dflat in pltdata])
            zmin = np.min([np.min([np.min(df) for df in dflat]) for dflat in pltdata])
            
            # create initial object
            for points,dat in zip(pltpoints, pltdata[0]):
#                 print(points, dat, pltpoints, pltdata)    
                plot = ax.plot_trisurf(points[:,0], points[:,1], dat, cmap=plt.get_cmap('jet'), vmin=zmin, vmax=zmax)
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
                for points,dat in zip(pltpoints, data[framenum]):
                    plot = ax.plot_trisurf(points[:,0], points[:,1], dat, linewidth=0, cmap=plt.get_cmap('jet'), vmin=zmin, vmax=zmax)
                ax.set_xlim3d(bbox.bounds[0])
                ax.set_xlabel('$s_1$')
                ax.set_ylim3d(bbox.bounds[1])
                ax.set_ylabel('$s_2$')
                ax.set_zlim3d([-0.5, zmax])
                ax.set_zlabel('$x$')
                ax.set_title('pdf animation test')
                return plot
       
            # Creating the Animation object
            pdf_ani = animation.FuncAnimation(fig, update_plot, frames, fargs=(pltdata, plot),
                                              interval=interval, blit=False)
            if directory is not None:
                pdf_ani.save('{}animation_{}.mp4'.format(directory, r), 
                             extra_args=['-vcodec', 'libx264'])
            if show:
                plt.show()
        except AttributeError: pass


