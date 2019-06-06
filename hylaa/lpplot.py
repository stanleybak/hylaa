'''
Stanley Bak
Functions related to plotting a set of lp constriants

get_verts is probably the main one to use

make_plot_vecs is useful for controlling the accuracy (and decreasing overhead compared w/ not passing it to get_verts)
'''

import math
import numpy as np
from scipy.spatial import ConvexHull

import hylaa.kamenev as kamenev
from hylaa.timerutil import Timers

def pt_to_plot_xy(pt, xdim=0, ydim=1, cur_time=0.0):
    '''convert a point to an x/y pair for plotting
    xdim and ydim can be either an integer (dimension number), an np.array (direction), or None (time will be used)
    '''

    x = y = float(cur_time)

    if isinstance(xdim, int):
        x = pt[xdim]
    elif xdim:
        assert isinstance(xdim, np.ndarray)
        x = np.dot(pt, xdim)

    if ydim and isinstance(ydim, int):
        y = pt[ydim]
    elif ydim:
        assert isinstance(ydim, np.ndarray)
        y = np.dot(pt, ydim)

    return x, y

def get_verts(lpi, xdim=0, ydim=1, plot_vecs=None, cur_time=0.0):
    '''get the vertices defining (an underapproximation) of the outside of the given linear constraints
    These will be usable for plotting, so that rv[0] == rv[-1]. A single point may be returned if the constraints
    are (close to) a single point.

    xdim and ydim can be either an integer (dimension number), an np.array (direction), or None (time will be used)

    cur_time can be an integer or a list of length 2 (min/max time)

    plot_vecs is an ordered list of vectors defining all the 2-d directions to optimize in... if None will 
    construct and use 256 equally spaced vectors 
    '''

    tol = 1e-9
    
    if plot_vecs is None:
        plot_vecs = make_plot_vecs()

    # make cur_time a 2-tuple: (min_time, max time)
    if isinstance(cur_time, (float, int)):
        cur_time = [float(cur_time), float(cur_time)]

    #try:
    if xdim is None and ydim is None:
        if abs(cur_time[0] - cur_time[1]) < tol:
            pts = [[cur_time, cur_time]]
        else:
            pts = []
            pts.append([cur_time.min, cur_time.min])
            pts.append([cur_time.min, cur_time.max])
            pts.append([cur_time.max, cur_time.max])
            pts.append([cur_time.max, cur_time.min])

    elif xdim is None:
        # plot over time
        if isinstance(ydim, int):
            ydim = np.array([1.0 if dim == ydim else 0.0 for dim in range(lpi.dims)], dtype=float)

        lpi.set_minimize_direction(ydim)
        res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])
        ymin = np.dot(ydim, res)

        lpi.set_minimize_direction(-1 * ydim)
        res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])
        ymax = np.dot(ydim, res)

        verts = [[cur_time[0], ymin]]

        if abs(ymin - ymax) > tol:
            verts.append([cur_time[0], ymax])

            if abs(cur_time[0] - cur_time[1]) > tol:
                verts.append([cur_time[1], ymax])

        if abs(cur_time[0] - cur_time[1]) > tol:
            verts.append([cur_time[1], ymin])

    elif ydim is None:
        # plot over time
        if isinstance(xdim, int):
            xdim = np.array([1.0 if dim == xdim else 0.0 for dim in range(lpi.dims)], dtype=float)

        lpi.set_minimize_direction(xdim)
        res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])
        xmin = np.dot(xdim, res)

        lpi.set_minimize_direction(-1 * xdim)
        res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])
        xmax = np.dot(xdim, res)

        verts = [[xmin, cur_time[0]]]

        if abs(xmin - xmax) > tol:
            verts.append([xmax, cur_time[0]])

            if abs(cur_time[0] - cur_time[1]) > tol:
                verts.append([xmax, cur_time[1]])

        if abs(cur_time[0] - cur_time[1]) > tol:
            verts.append([xmin, cur_time[1]])
    else:
        # 2-d plot
        bboxw = bbox_widths(lpi, xdim, ydim)

        # use bbox to determine acceptable accuracy... might be better to somehow do this
        epsilon = min(bboxw) / 1000.0
        dim_list = [xdim, ydim]

        def supp_point_nd(vec):
            'return a supporting point for the given direction (maximize)'

            assert len(vec) == len(dim_list)

            d = np.zeros((lpi.dims,), dtype=float)
            # negative here because we want to MAXIMIZE not minimize

            for i, dim_index in enumerate(dim_list):
                d[dim_index] = -vec[i]

            lpi.set_minimize_direction(d)

            res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(lpi.dims)])

            rv = []

            for dim in dim_list:
                rv.append(res[dim])

            rv = np.array(rv, dtype=float)

            return rv

        Timers.tic('kamenev.get_verts')
        verts = kamenev.get_verts(len(dim_list), supp_point_nd, epsilon=epsilon)
        Timers.toc('kamenev.get_verts')

        if len(verts) > 2:
            # make 2-d convex hull to fix the order
            hull = ConvexHull(verts)

            verts = [[verts[i, 0], verts[i, 1]] for i in hull.vertices]

        # old method
        #pts = find_boundary_pts(lpi, xdim, ydim, plot_vecs, bboxw)
        #verts = [[pt[0], pt[1]] for pt in pts]

    # wrap polygon back to first point
    verts.append(verts[0])

    return verts

def bbox_widths(lpi, xdim, ydim):
    'find and return the bounding box widths of the lp for the passed-in dimensions'

    rv = []
    dims = lpi.dims

    for dim in [xdim, ydim]:
        col = lpi.cur_vars_offset + dim
        min_dir = [1 if i == dim else 0 for i in range(dims)]
        max_dir = [-1 if i == dim else 0 for i in range(dims)]
        
        min_val = lpi.minimize(direction_vec=min_dir, columns=[col])[0]
        max_val = lpi.minimize(direction_vec=max_dir, columns=[col])[0]

        dx = max_val - min_val

        if dx < 1e-5:
            dx = 1e-5

        rv.append(dx)

    return rv

def make_plot_vecs(num_angles=256, offset=0.0):
    'make plot_vecs with equally spaced directions, returning the result'

    plot_vecs = []

    for theta in np.linspace(0.0, 2.0*math.pi, num_angles, endpoint=False):
        x = math.cos(theta + offset)
        y = math.sin(theta + offset)

        vec = np.array([x, y], dtype=float)

        plot_vecs.append(vec)

    return plot_vecs

def find_boundary_pts(lpi, xdim, ydim, plot_vecs, bboxw):
    '''
    find points along an LPs boundary by solving several LPs and
    returns a list of points on the boundary which maximize each
    of the passed-in directions
    '''

    rv = []

    assert len(plot_vecs) >= 2

    # optimized approach: do binary search to find changes
    point = _minimize(lpi, xdim, ydim, plot_vecs[0], bboxw)
    rv.append(point.copy())

    # add it in thirds, to ensure we don't miss anything
    third = len(plot_vecs) // 3

    # 0 to 1/3
    point = _minimize(lpi, xdim, ydim, plot_vecs[third], bboxw)

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, 0, third, rv[-1], point, xdim, ydim, plot_vecs, bboxw)
        rv.append(point.copy())

    # 1/3 to 2/3
    point = _minimize(lpi, xdim, ydim, plot_vecs[2*third], bboxw)

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, third, 2*third, rv[-1], point, xdim, ydim, plot_vecs, bboxw)
        rv.append(point.copy())

    # 2/3 to end
    point = _minimize(lpi, xdim, ydim, plot_vecs[-1], bboxw)

    if not np.array_equal(point, rv[-1]):
        rv += _binary_search_boundaries(lpi, 2*third, len(plot_vecs) - 1, rv[-1], point, xdim, ydim, plot_vecs, bboxw)
        rv.append(point.copy())

    # pop last point if it's the same as the first point
    if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
        rv.pop()

    return rv

def _minimize(lpi, xdim, ydim, direction, bounding_box_widths):
    'minimize to lp... returning the 2-d point of the minimum'

    dims = 0

    assert not (xdim is None and ydim is None)

    if xdim is not None:
        dims = xdim + 1 if isinstance(xdim, int) else len(xdim)

    if ydim is not None:
        dims = max(dims, ydim + 1) if isinstance(ydim, int) else max(dims, len(ydim))
        
    if isinstance(xdim, int):
        xdim = np.array([1.0 if dim == xdim else 0.0 for dim in range(dims)], dtype=float)

    if isinstance(ydim, int):
        ydim = np.array([1.0 if dim == ydim else 0.0 for dim in range(dims)], dtype=float)

    # xdim and ydim are both arrays or None
    optimize_direction = np.zeros(dims, dtype=float)

    if xdim is not None:
        optimize_direction += direction[0] * xdim / bounding_box_widths[0]

    if ydim is not None:
        optimize_direction += direction[1] * ydim / bounding_box_widths[1]

    lpi.set_minimize_direction(optimize_direction)

    res = lpi.minimize(columns=[lpi.cur_vars_offset + n for n in range(dims)])

    xcoord = 0 if xdim is None else np.dot(res, xdim)
    ycoord = 0 if ydim is None else np.dot(res, ydim)
    
    return np.array([xcoord, ycoord], dtype=float)

def _binary_search_boundaries(lpi, start, end, start_point, end_point, xdim, ydim, plot_vecs, bboxw):
    '''
    return all the optimized points in the star for the passed-in directions, between
    the start and end indices, exclusive

    points which match start_point or end_point are not returned
    '''

    rv = []

    if start + 1 < end:
        mid = (start + end) // 2

        mid_point = _minimize(lpi, xdim, ydim, plot_vecs[mid], bboxw)

        not_start = not np.allclose(start_point, mid_point, atol=1e-6)
        not_end = not np.allclose(end_point, mid_point, atol=1e-6)

        if not_start:
            rv += _binary_search_boundaries(lpi, start, mid, start_point, mid_point, xdim, ydim, plot_vecs, bboxw)

        if not_start and not_end:
            rv.append(mid_point)

        if not_end:
            rv += _binary_search_boundaries(lpi, mid, end, mid_point, end_point, xdim, ydim, plot_vecs, bboxw)

    return rv
