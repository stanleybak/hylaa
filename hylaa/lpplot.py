'''
Stanley Bak
Functions related to plotting a set of lp constriants

get_verts is probably the main one to use

make_plot_vecs is useful for controlling the accuracy (and decreasing overhead compared w/ not passing it to get_verts)
'''

import math
import numpy as np

def get_verts(lpi, num_dims, xdim=0, ydim=1, plot_vecs=None):
    '''get the vertices defining (an underapproximation) of the outside of the given linear constraints
    These will be usable for plotting, so that rv[0] == rv[-1]

    plot_vecs is an ordered list of vectors defining all the 2-d directions to optimize in... if None will 
    construct and use 256 equally spaced vectors 
    '''
    
    if plot_vecs is None:
        plot_vecs = make_plot_vecs(256)

    # first set the optimization direction to all zeros
    zero_vec = np.zeros((num_dims,), dtype=float)
    lpi.set_minimize_direction(zero_vec)

    pts = _find_boundary_pts(lpi, xdim, ydim, plot_vecs)

    verts = [[pt[0], pt[1]] for pt in pts]

    # wrap polygon back to first point
    verts.append(verts[0])

    return verts

def make_plot_vecs(num_angles=256):
    'make plot_vecs with equally spaced directions, returning the result'

    plot_vecs = []

    step = 2.0 * math.pi / num_angles

    for theta in np.arange(0.0, 2.0*math.pi, step):
        x = math.cos(theta)
        y = math.sin(theta)

        vec = np.array([x, y], dtype=float)

        plot_vecs.append(vec)

    return plot_vecs

def _find_boundary_pts(lpi, xdim, ydim, plot_vecs):
    '''
    find points along an LPs boundary by solving several LPs and
    returns a list of points on the boundary which maximize each
    of the passed-in directions
    '''

    rv = []

    assert len(plot_vecs) >= 2

    # optimized approach: do binary search to find changes
    point = _minimize(lpi, xdim, ydim, plot_vecs[0])
    rv.append(point.copy())

    # add it in thirds, to ensure we don't miss anything
    third = len(plot_vecs) // 3

    # 0 to 1/3
    point = _minimize(lpi, xdim, ydim, plot_vecs[third])

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, 0, third, rv[-1], point, xdim, ydim, plot_vecs)
        rv.append(point.copy())

    # 1/3 to 2/3
    point = _minimize(lpi, xdim, ydim, plot_vecs[2*third])

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, third, 2*third, rv[-1], point, xdim, ydim, plot_vecs)
        rv.append(point.copy())

    # 2/3 to end
    point = _minimize(lpi, xdim, ydim, plot_vecs[-1])

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, 2*third, len(plot_vecs) - 1, rv[-1], point, xdim, ydim, plot_vecs)
        rv.append(point.copy())

    # pop last point if it's the same as the first point
    if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
        rv.pop()

    return rv

def _minimize(lpi, xdim, ydim, direction):
    'minimize to lp... returning the 2-d point of the first 2 coordinates of the minimum'

    assert isinstance(xdim, int)
    assert isinstance(ydim, int), "ydim was {}".format(ydim)

    dir_vec = [direction[0] if dim == xdim else direction[1] if dim == ydim else 0 for dim in range(max(xdim, ydim) + 1)]
    lpi.set_minimize_direction(dir_vec)

    return lpi.minimize_partial_result([xdim, ydim])

def _binary_search_boundaries(lpi, start, end, start_point, end_point, xdim, ydim, plot_vecs):
    '''
    return all the optimized points in the star for the passed-in directions, between
    the start and end indices, exclusive

    points which match start_point or end_point are not returned
    '''

    rv = []

    if start + 1 < end:
        mid = (start + end) // 2

        mid_point = _minimize(lpi, xdim, ydim, plot_vecs[mid])

        not_start = not np.allclose(start_point, mid_point, atol=1e-3)
        not_end = not np.allclose(end_point, mid_point, atol=1e-3)

        if not_start:
            rv += _binary_search_boundaries(lpi, start, mid, start_point, mid_point, xdim, ydim, plot_vecs)

        if not_start and not_end:
            rv.append(mid_point)

        if not_end:
            rv += _binary_search_boundaries(lpi, mid, end, mid_point, end_point, xdim, ydim, plot_vecs)

    return rv