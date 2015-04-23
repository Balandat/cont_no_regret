'''
A collection of Domain classes for the Continuous No Regret Problem.  

@author: Maximilian Balandat, Walid Krichene
@date: Feb 12, 2015
'''

import numpy as np
from numpy import Inf
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
    """ Base class for domains """
    
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
