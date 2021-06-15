'''
Functions related to Kamenev's method for polytope approximation (the method of refined bounds)

Stanley Bak
May 16, 2019
'''

import math

import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull

from hylaa.timerutil import Timers

def _get_orthonormal_rank(vecs, tol=1e-7):
    '''
    given a list of vecs, return a new vector orthonormal to them and the rank of the matrix
    '''

    _, s, v = np.linalg.svd(vecs)

    index = 0

    while index < len(s) and s[index] > tol:
        index += 1

    if index == len(v):
        rv_vec = None # the vectors span the space
    else:
        rv_vec = v[index]

    return rv_vec, index

def _get_rank(vecs, tol=1e-7):
    '''get the rank of the passed in matrix'''

    return _get_orthonormal_rank(vecs, tol=tol)[1]

def _find_two_points(dims, supp_point_func):
    '''find two points in the the convex set defined through supp_point_func (which may be degenerate)

    if len(pts) == 1, the convex set is a degenerate set consisting of a single pt
    '''

    pts = []

    for d in range(dims):
        vec = np.array([-1 if i == d else 0 for i in range(dims)], dtype=float)

        # try min
        p1 = supp_point_func(vec)
        assert len(p1) == dims, f"support fuction returned {len(p1)}-dimensional point, expected {dims}-d"

        pts = [p1]

        # try max
        vec = np.array([1 if i == d else 0 for i in range(dims)], dtype=float)
        p2 = supp_point_func(vec)

        if not np.allclose(p1, p2):
            pts = [p1, p2]
            break

    return pts

def _find_init_simplex(dims, supp_point_func):
    '''
    find an n-dimensional initial simplex
    '''

    Timers.tic('init_simplex')

    # first, construct the initial simplex and determine a basis for the convex set (it may be degenerate)
    init_simplex = _find_two_points(dims, supp_point_func)

    if len(init_simplex) == 2: # S may be a degenerate shape consisting of a single point
        init_vec = init_simplex[1] - init_simplex[0]

        spanning_dirs = [init_vec]
        degenerate_dirs = []
        vecs = [init_vec]

        for _ in range(dims - 1):
            new_dir, rank = _get_orthonormal_rank(vecs)

            # min/max in direction v, checking if it increases the rank of vecs
            pt = supp_point_func(new_dir)
            vecs.append(pt - init_simplex[0])

            if _get_rank(vecs) > rank:
                init_simplex.append(pt)
                spanning_dirs.append(vecs[-1])
                continue

            # rank did not increase with maximize, try minimize
            vecs = vecs[0:-1] # pop vec

            pt = supp_point_func(-1 * new_dir)
            vecs.append(pt - init_simplex[0])

            if _get_rank(vecs) > rank:
                init_simplex.append(pt)
                spanning_dirs.append(vecs[-1])
                continue

            # rank still didn't increase, new_dir is orthogonal to shape S
            vecs = vecs[0:-1] # pop vec

            vecs.append(new_dir) # forces a new orthonormal direction during the next iteration
            degenerate_dirs.append(new_dir)

    Timers.toc('init_simplex')

    return init_simplex

def get_verts(dims, supp_point_func, epsilon=1e-7):
    '''
    get the n-dimensional vertices of the convex set defined through supp_point_func (which may be degenerate)
    '''

    init_simplex = _find_init_simplex(dims, supp_point_func)

    if len(init_simplex) < 3:
        return init_simplex # for 0-d and 1-d sets, the init_simplex corners are the only possible extreme points
    
    rv, _ = _v_h_rep_given_init_simplex(init_simplex, supp_point_func, epsilon=epsilon)

    return rv

def _v_h_rep_given_init_simplex(init_simplex, supp_point_func, epsilon=1e-7):
    '''get all the vertices and hyperplanes of (an epsilon approximation of) the set, defined through supp_point_func

    This function is provided with an initial simplex which spans the space

    this returns verts, equations, where equations is from the Convex Hull's (hull.equations)
    '''

    new_pts = init_simplex
        
    verts = []
    iteration = 0
    max_error = None

    while new_pts:
        iteration += 1
        #print(f"\nIteration {iteration}. Verts: {len(verts)}, new_pts: {len(new_pts)}, max_error: {max_error}")
                
        first_new_index = len(verts)
        verts += new_pts
        new_pts = []
        max_error = 0

        Timers.tic('ConvexHull')
        hull = ConvexHull(verts)
        Timers.toc('ConvexHull')

        for i, simplex in enumerate(hull.simplices):
            is_new = False

            for index in simplex:
                if index >= first_new_index:
                    is_new = True
                    break

            if not is_new:
                continue # skip this simplex

            # get hyperplane for simplex
            normal = hull.equations[i, :-1]
            rhs = -1 * hull.equations[i, -1]

            Timers.tic('supp_point_func')
            supporting_pt = supp_point_func(normal)
            Timers.toc('supp_point_func')
            
            error = np.dot(supporting_pt, normal) - rhs
            max_error = max(max_error, error)

            if error <= -1e-4:
                print(f"Kamenev Plot warning: supporting point was inside facet? error was {error}")

            if error >= epsilon:
                # add the point
                already_added = False

                Timers.tic("check if new point")
                for pt in new_pts:
                    if np.allclose(pt, supporting_pt):
                        already_added = True
                        break
                Timers.toc("check if new point")

                if not already_added:
                    new_pts.append(supporting_pt)

    return np.array(verts, dtype=float), hull.equations
