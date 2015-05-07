'''
A collection of Domain classes for the Continuous No Regret Problem.  

@author: Maximilian Balandat
@date: May 6, 2015
'''

import numpy as np
from scipy.misc import factorial
from cvxopt import solvers, matrix

class Domain(object):
    """ Base class for domains """
        
    def iselement(self, points):
        """ Returns boolean array of the same dimension as points. 
            "True" if point is contained in domain and "False" otherwise """
        raise NotImplementedError

    def compute_parameters(self):
        """ Computes diameter and volume of the domain for later use """
        raise NotImplementedError
    
    def bbox(self):
        """ Computes a Bounding Box (a Rectangle object) """
        raise NotImplementedError
    
    def union(self, domain):
        """ Returns a Domain object that is the union of the original 
        object with dom (also a Domain object) """
    
    def isconvex(self):
        """ Returns True if the domain is convex, and False otherwise """
        return self.cvx
    
    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2 """
        raise NotImplementedError
    
    
class Rectangle(Domain):
    """ A rectangle (a rectangular box in 2 dimensions) """
    
    def __init__(self, xlim, ylim):
        """ Constructor """
        self.n = 2
        self.cvx = True
        self.lb = np.array([xlim[0], ylim[0]])
        self.ub = np.array([xlim[1], ylim[1]])
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.all(self.lb<=points, axis=1)*np.all(self.ub >= points, axis=1)

    def compute_parameters(self):
        """ Computes diameter, volume and uniformity parameter v of the domain for later use """
        diameter = np.sqrt((self.ub[0]-self.lb[0])**2 + (self.ub[1]-self.lb[1])**2)
        volume = (self.ub[0]-self.lb[0])*(self.ub[1]-self.lb[1])
        return diameter, volume, 1.0
    
    def bbox(self):
        """ A Rectangle is its own Bounding Box """
        return self
    
    def grid(self, N):
        """ Returns a uniform grid with at least N gridpoints """
        n = np.ceil(np.sqrt(N))
        xpoints = np.linspace(self.lb[0], self.ub[0], n)
        ypoints = np.linspace(self.lb[1], self.ub[1], n)
        return np.vstack(np.meshgrid(xpoints,ypoints)).reshape(2,-1).T
    
    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        # for a rectangle this is easy, we just need to check the vertices
        vertices = np.array([[self.lb[0], self.lb[1]], [self.lb[0], self.ub[1]], 
                             [self.ub[0], self.lb[1]], [self.ub[0], self.ub[1]]]) 
        return np.max([np.linalg.norm(mu - vertex) for vertex in vertices])
    
    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2. For a rectangle this 
            is straightforward and can be done efficiently. """
        xs = (self.lb[0]*(self.lb[0] > points[:,0]) +
              self.ub[0]*(self.ub[0] < points[:,0]) + 
              points[:,0]*((self.lb[0] <= points[:,0]) & (points[:,0] <= self.ub[0])))
        ys = (self.lb[1]*(self.lb[1] > points[:,1]) +
              self.ub[1]*(self.ub[1] < points[:,1]) + 
              points[:,1]*((self.lb[1] <= points[:,1]) & (points[:,1] <= self.ub[1])))
        return np.array([xs,ys]).T
    
    def gen_project(self, points, mats):
        """ Performs a generalized projection of the points in 'points' onto the
            domain, returning x* = argmin_{s \in domain} (s-x)^T*A*(s-x). """
        contained = self.iselement(points)
        proj_points, dists = [], []
        for point, mat, inint in zip(points, mats, contained):
            if inint:
                proj_points.append(point)
                dists.append(0)
            else:
                solvers.options['show_progress'] = False
                Acvx = matrix(2*mat, tc='d')
                bcvx = matrix(-2*np.dot(mat, point), tc='d')
                Gcvx = matrix([[1,-1,0,0], [0,0,1,-1]], tc='d')
                lb, ub = self.lb.astype(float), self.ub.astype(float)
                hcvx = matrix([ub[0], -lb[0], ub[1], -lb[1]], tc='d')
                res = solvers.qp(Acvx, bcvx, Gcvx, hcvx)
                proj_points.append(np.array(res['x']).flatten())
                dists.append(res['primal objective'] + np.dot(point, np.dot(mat, point)))
        return np.array(proj_points), np.array(dists)
        
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the rectangle """
        xsamples = np.random.uniform(low=self.lb[0], high=self.ub[0], size=(N,1))
        ysamples = np.random.uniform(low=self.lb[1], high=self.ub[1], size=(N,1))
        return np.concatenate((xsamples, ysamples), axis=1)
    
    
class UnionOfDisjointRectangles(Domain):
    """ Domain consisting of a number of p/w disjoint rectangles """
    
    def __init__(self, rects):
        """ Constructor """
        self.rects = rects
        self.cvx = False
        self.n = rects[0].n    
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.any([rect.iselement(points) for rect in self.rects], axis=0)

    def compute_parameters(self):
        """ Computes diameter and volume of the domain as well as a lower bound of  
            the uniform fatness constant v for later use (the bound on v is tight 
            if any two rectangles are a strictly positive distance apart). """
        # since rectangles are disjoint, the volume is just the sum of the 
        # individual volumes and v is the minimum of the individual vs
        bbox = self.bbox()
        bbox_diameter = bbox.compute_parameters()[0]
        volumes = [rect.volume for rect in self.rects]
        volume = np.sum(volumes)
        return bbox_diameter, volume, np.min(volumes)/volume
    
    def set_v(self, v):
        """ Manually sets the uniformity parameter v. This is useful if 
            the value is known but numerical computation is very clumsy. """
        self.v = v
    
    def bbox(self):
        """ Returns the bounding box rectangle """
        lb = np.min([rect.lb for rect in self.rects], axis=0)
        ub = np.max([rect.ub for rect in self.rects], axis=0)
        return Rectangle((lb[0], ub[0]), (lb[1], ub[1]))
    
    def grid(self, N):
        """ Returns a uniform grid with at least N grid points """
        volumes = np.array([rect.volume for rect in self.rects])
        weights = volumes/np.sum(volumes)
        return np.vstack([rect.grid(weight*N) for rect,weight 
                          in zip(self.rects, weights)])
    
    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        # for a union of disjoint rectangles it suffices to check 
        # the vertices of each rectangle
        D = 0
        for rect in self.rects:   
            vertices = np.array([[rect.lb[0], rect.lb[1]], [rect.lb[0], rect.ub[1]], 
                                 [rect.ub[0], rect.lb[1]], [rect.ub[0], rect.ub[1]]]) 
            D = max(D, np.max([np.linalg.norm(mu - vertex) for vertex in vertices]))
        return D
    
    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2. For a union of disjoint 
            rectangle we can just project on each rectangle and take the point
            with the minimum distance. """
        proj_points = np.array([rect.project(points) for rect in self.rects])
        dists = np.array([np.linalg.norm(points - proj, axis=1) for proj in proj_points])
        return proj_points[np.argmin(dists, 0), np.arange(proj_points.shape[1])]
    
    def gen_project(self, points, mats):
        """ Performs a generalized projection of the points in 'points' onto the
            domain, returning x* = argmin_{s \in domain} (s-x)^T*A*(s-x). 
            For a union of disjoint convex sets we project on each set and take 
            the point with the minimum distance. """
        proj_points, dists = [], []
        for rect in self.rects:
            ppts, dsts = rect.gen_project(points, mats)
            proj_points.append(ppts)
            dists.append(dsts)
        proj_points = np.array(proj_points)
        dists = np.array(dists)
        return proj_points[np.argmin(dists, 0), np.arange(proj_points.shape[1])], np.min(dists, 0)
        
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the union of disjoint rectangles """
        volumes = [rect.volume for rect in self.rects]
        weights = volumes/np.sum(volumes)
        select = np.random.choice(np.arange(len(volumes)), p=weights, size=N)
        samples = np.array([rect.sample_uniform(N) for rect in self.rects])
        return samples[select, np.arange(N)]
    
    
class nBox(Domain):
    """ A rectangular box in n dimensions """
    
    def __init__(self, bounds):
        """ Construct an nBox. bounds is a list of n tuples, where the i-th tuple
            gives the lower and upper bound of the box in dimension i """
        self.n = len(bounds)
        self.cvx = True
        self.verts = None
        self.bounds = bounds
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.all(np.array([(points[:,i]>=self.bounds[i][0]) & 
                                (points[:,i]<=self.bounds[i][1]) 
                                for i in range(self.n)]), axis=0)

    def compute_parameters(self):
        """ Computes diameter, volume and uniformity parameter v of the domain for later use """
        diameter = np.sqrt(np.sum([(self.bounds[i][1] - self.bounds[i][0])**2 for i in range(self.n)]))
        volume = np.array([self.bounds[i][1] - self.bounds[i][0] for i in range(self.n)]).prod()
        return diameter, volume, 1.0
    
    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        return np.max([np.linalg.norm(mu - vertex) for vertex in self.vertices()])
    
    def bbox(self):
        """ An nBox is its own Bounding Box """
        return self
    
    def vertices(self):
        """ Returns the vertices of the nBox """
        if self.verts is None:
            bounds = [np.asarray(bnd) for bnd in self.bounds]
            shape = (len(arr) for arr in bounds)
            idxs = np.indices(shape, dtype=int)
            idxs = idxs.reshape(len(bounds), -1).T
            self.verts = np.array([bnd[idxs[:, n]] for n,bnd in enumerate(bounds)]).T
        return self.verts    
    
    def grid(self, N):
        """ Returns a uniform grid with at least N gridpoints """
        Z = (np.prod([self.bounds[i][1] - self.bounds[i][0] for i in range(self.n)]))**(1/self.n)
        Ns = [np.ceil((self.bounds[i][1] - self.bounds[i][0])/Z*N**(1/self.n)) for i in range(self.n)]
        grids = [np.linspace(self.bounds[i][0], self.bounds[i][1], Ns[i]) for i in range(self.n)]
        return np.vstack(np.meshgrid(*grids)).reshape(self.n,-1).T
        
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the nBox """
        return np.concatenate([np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1], 
                                                 size=(N,1)) for i in range(self.n)], axis=1)
    
  
class UnionOfDisjointnBoxes(Domain):
    """ Domain consisting of a number of p/w disjoint nBoxes """
    
    def __init__(self, nboxes):
        """ Constructor """
        self.nboxes = nboxes
        self.cvx = False
        self.verts = None
        self.n = nboxes[0].n    
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.any([nbox.iselement(points) for nbox in self.nboxes], axis=0)

    def compute_parameters(self):
        """ Computes diameter and volume of the domain as well as a lower bound of  
            the uniform fatness constant v for later use (the bound on v is tight 
            if any two rectangles are a strictly positive distance apart). """
        # since rectangles are disjoint, the volume is just the sum of the 
        # individual volumes and v is the minimum of the individual vs
        bbox = self.bbox()
        bbox_diameter = bbox.compute_parameters()[0]
        volumes = [nbox.volume for nbox in self.nboxes]
        volume = np.sum(volumes)
        return bbox_diameter, volume, np.min(volumes)/volume
    
    def bbox(self):
        """ Returns the bounding nBox """
        lower = np.array([[self.nboxes[i].bounds[j][0] for j in range(self.n)] 
                          for i in range(len(self.nboxes))]).min(axis=0)
        upper = np.array([[self.nboxes[i].bounds[j][1] for j in range(self.n)] 
                          for i in range(len(self.nboxes))]).max(axis=0)
        bounds = [(low, high) for low,high in zip(lower, upper)]
        return nBox(bounds)
    
    
    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        return np.max([np.linalg.norm(mu - vertex) for vertex in self.vertices()])
    
    def vertices(self):
        """ Returns the vertices of the UnionOfDisjointnBoxes """
        if self.verts is None:
            self.verts = np.vstack([nbox.vertices() for nbox in self.nboxes])
        return self.verts   
    
    def grid(self, N):
        """ Returns a uniform grid with at least N grid points """
        volumes = np.array([nbox.volume for nbox in self.nboxes])
        weights = volumes/np.sum(volumes)
        return np.vstack([nbox.grid(weight*N) for nbox,weight 
                          in zip(self.nboxes, weights)])
        
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the union of disjoint nBoxes """
        volumes = [nbox.volume for nbox in self.nboxes]
        weights = volumes/np.sum(volumes)
        select = np.random.choice(np.arange(len(volumes)), p=weights, size=N)
        samples = np.array([nbox.sample_uniform(N) for nbox in self.nboxes])
        return samples[select, np.arange(N)]
    
    
class DifferenceOfnBoxes(Domain):
    """ Domain consisting of an nBox (the 'outer' box) from which 
        multiple (disjoint) nBoxes are subtracted """
    
    def __init__(self, outer, inner):
        """ Constructor """
        violations = [((obnd[0]>ibnd[1]) | (obnd[1]<ibnd[1])) 
                      for nbox in inner for obnd,ibnd in zip(outer.bounds, nbox.bounds)]
        if np.any(violations):
            raise Exception('All nBoxes in "inner" must be contained inside the outer nBox')
        self.outer = outer
        self.inner = inner
        self.cvx = False
        self.verts = None
        self.n = outer.n    
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        in_inner = np.any([nbox.iselement(points) for nbox in self.inner], axis=0)
        return self.outer.iselement(points) & np.logical_not(in_inner)
   
    def compute_parameters(self):
        """ Computes diameter and volume of the domain as well as a lower bound of  
            the uniform fatness constant v for later use (the bound on v is tight 
            if any two rectangles are a strictly positive distance apart). """
        # since the inner rectangles are disjoint, the volume is just the volume of
        # the outer nBox minus the sum of the volumes of the inner nBoxes
        # Unfortunately determining v is more difficult ...
        diameter, volume_outer = self.outer.compute_parameters()[0:2]
        volume = volume_outer - np.sum([nbox.volume for nbox in self.inner])
        return diameter, volume, None
    
    def bbox(self):
        """ Returns the bounding nBox, which is just the outer nBox. """ 
        return self.outer    
    
    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2. Here it suffices to 
            just check the vertices of the outer nBox. """
        return self.outer.compute_Dmu(mu)
    
    def vertices(self):
        """ Returns an array with the the vertices of the DifferenceOfnBoxes.
            This is just the union of the set of vertices from the individual nBoxes. """
        return np.vstack([self.outer.vertices(), np.vstack([nbox.vertices() for nbox in self.inner])])
    
    def grid(self, N):
        """ Returns a uniform grid with at least N grid points. This implementation is
            based on inflating the number of gridpoints for the grid on the outer nBox, 
            and then returning only points that do not fall inside the inner nBoxes. """
        full_grid = self.outer.grid(self.outer.volume/self.volume*N)
        return full_grid[self.iselement(full_grid)]
        
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the DifferenceOfnBoxes. This can be achieved 
            by simple rejection sampling, i.e. by sampling uniformly from the outer nBox 
            and discarding samples tha fall inside the inner nBoxes. This is obviously 
            not very efficient and should be avoided if an additive decomposition
            of the DifferenceOfnBoxes is available. """
        samples_outer = self.outer.sample_uniform(self.outer.volume/self.volume*N)
        samples = samples_outer[self.iselement(samples_outer)]
        while len(samples) < N:
            newsamples = self.outer.sample_uniform(N/(N-len(samples))*self.outer.volume/self.volume*N)
            samples = np.vstack((samples, newsamples[self.iselement(newsamples)]))
        return samples[0:N]

    
class UnitSimplex(Domain):
    """ The k-unit simplex (i.e. x_0,x_1,...,x_k s.t. x_i>=0 and sum(x_i)=1) """
    
    def __init__(self, k):
        self.k, self.n = k, k+1
        self.cvx = True
        self.diameter, self.volume, self.v = self.compute_parameters()
        
    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.all(0<=points, axis=1)*np.all(points<=1, axis=1)*(np.sum(points, axis=1) == True)

    def compute_parameters(self):
        """ Computes diameter, volume and uniformity parameter v of the domain for later use """
        return np.sqrt(self.n), 1/factorial(self.k), 1.0
    
    def bbox(self):
        """ The bounding box of a k-unit simplex is the (k+1) unit cube """
        return nBox([(0,1)]*self.n)
    
    def grid(self, N):
        """ Returns a uniform grid with at least N gridpoints """
        # this does not seem super straightforward
        return NotImplementedError
           
    def sample_uniform(self, N):
        """ Draws N samples uniformly from the k-unit simplex. This is just
            sampling from a Dirichlet distribution with all parameters equal to 1. """
        return np.random.dirichlet([1]*self.k, N)
    

      
def CAL():
    """ Helper function to create the 'CAL' set """
    xlims = [[0.0, 2.5], [0.0, 1.0], [0.0, 2.5],
             [3.0, 4.0], [3.0, 6.0], [5.0, 6.0], [4.0, 5.0],
             [6.5, 7.5], [6.5, 9.0]]
    ylims = [[0.0, 1.0], [1.0, 3.0], [3.0, 4.0],
             [0.0, 4.0], [3.0, 4.0], [0.0, 4.0], [1.0, 2.0],
             [1.0, 4.0], [0.0, 1.0]]
    C = UnionOfDisjointRectangles([Rectangle(xlim, ylim) for xlim,ylim in zip(xlims, ylims)])
    C.set_v(2.5/C.volume)
    return C

def S():
    """ Helper function to create the 'S' set """
    xlims = [[0.0, 3.0], [2.0, 3.0], [0.0, 2.0], [0.0, 1.0], [1.0, 3.0]]
    ylims = [[0.0, 1.0], [1.0, 3.0], [2.0, 3.0], [3.0, 5.0], [4.0, 5.0]]
    S = UnionOfDisjointRectangles([Rectangle(xlim, ylim) for xlim,ylim in zip(xlims, ylims)])
    S.set_v(3.0/11)
    return S
