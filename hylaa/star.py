'''
Generalized Star and other Star data structures
Stanley Bak
Aug 2016
'''

import math
import time

import numpy as np
from numpy import array_repr
from numpy.linalg import lstsq
from numpy.testing import assert_array_almost_equal

import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix

from hylaa.glpk_interface import LpInstance
from hylaa.hybrid_automaton import HyperRectangle, LinearAutomatonTransition
from hylaa.hybrid_automaton import LinearAutomatonMode
from hylaa.timerutil import Timers as Timers
from hylaa.util import Freezable
from hylaa.settings import PlotSettings, HylaaSettings, TimeElapseSettings
from hylaa.time_elapse import TimeElapser
from hylaa.guard_opt_data import GuardOptData

class Star(Freezable):
    '''
    A representation of a set of continuous states. Contains logic
    for plotting that states if requested in the settings.
    '''

    def __init__(self, hylaa_settings, mode, init_space_csc, init_mat, init_rhs, init_range_tuples=None):
        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(mode, LinearAutomatonMode)
        assert isinstance(init_space_csc, csc_matrix)
        assert isinstance(init_mat, csr_matrix)
        assert isinstance(init_rhs, np.ndarray)

        self.mode = mode
        self.dims = mode.a_matrix_csr.shape[0]
        self.inputs = 0 if mode.b_matrix_csc is None else mode.b_matrix_csc.shape[1]

        init_rhs.shape = (len(init_rhs), ) # flatten init_rhs into a 1-d array
        assert init_rhs.shape == (init_mat.shape[0],)
        assert init_mat.shape[1] == init_space_csc.shape[1]
        assert init_space_csc.shape[0] == self.dims
        assert init_space_csc.shape[1] > 0, "initial states are zero-dimensional"

        self.num_init_vars = init_space_csc.shape[1]

        self.settings = hylaa_settings

        self.time_elapse = TimeElapser(mode, hylaa_settings, init_space_csc)

        self.init_space_csc = init_space_csc
        self.init_mat = init_mat
        self.init_rhs = init_rhs
        self.init_range_tuples = init_range_tuples

        ###################################
        ## private member initialization ##
        ###################################
        self._plot_lpi = None # LpInstance for plotting
        self._verts = None # for plotting optimization, a cached copy of this star's projected polygon verts

        self._guard_opt_data = []

        for i in xrange(len(mode.transitions)):
            self._guard_opt_data.append(GuardOptData(self, mode, i))

        self.num_plot_vars = 0
        for d in [self.settings.plot.xdim_dir, self.settings.plot.ydim_dir]:
            if d is not None:
                self.num_plot_vars += 1

        self.freeze_attrs()

    def get_guard_intersection(self, transition_index):
        '''update the LP for the given transition, solve, and return get the lp solution (if feasible)'''

        return self._guard_opt_data[transition_index].get_updated_lp_solution()

    def get_guard_lpi(self, transition_index):
        '''update the LP for the given transition, solve, and return get the lp solution (if feasible)'''

        return self._guard_opt_data[transition_index].get_guard_lpi()

    def get_plot_lpi(self):
        'get (maybe create) the LpInstance object for this star, and return it'

        assert self.time_elapse.cur_basis_mat is not None

        rv = self._plot_lpi

        if rv is None:
            rv = LpInstance(self.num_plot_vars, self.num_init_vars, self.inputs)
            rv.set_init_constraints(self.init_mat, self.init_rhs)
            rv.set_no_output_constraints()

            rv.update_basis_matrix(self.time_elapse.cur_basis_mat[:self.num_plot_vars])

            if self.inputs > 0:
                rv.set_input_constraints_csc(csc_matrix(self.mode.u_constraints_csr), self.mode.u_constraints_rhs)

            self._plot_lpi = rv

        return rv

    def step(self):
        'update the star based on values from a new simulation time instant'

        self.time_elapse.step()

        if self._plot_lpi is not None:
            self._plot_lpi.update_basis_matrix(self.time_elapse.cur_basis_mat[:self.num_plot_vars])

            if self.time_elapse.cur_input_effects_matrix is not None:
                self._plot_lpi.add_input_effects_matrix(self.time_elapse.cur_input_effects_matrix[:2])

        self._verts = None # cached vertices for plotting are no longer valid

    ######### star plotting methods below ############

    # global
    plot_vecs = None # list of vectors to optimize in for plotting, assigned in Star.init_plot_vecs
    plot_settings = None # assigned in Star.init_plot_vecs
    high_vert_mode = False # reduce plotting directions if the set has lots of verticies (drawing optimization)

    @staticmethod
    def init_plot_vecs(plot_settings):
        'initialize plot_vecs'

        num_plot_vars = 0
        for d in [plot_settings.xdim_dir, plot_settings.ydim_dir]:
            if d is not None:
                num_plot_vars += 1

        assert num_plot_vars > 0, "both plot.xdim_dir and plot.ydim_dir are None: not allowed"

        Star.plot_settings = plot_settings
        Star.plot_vecs = []

        if num_plot_vars == 1:
            # single dimensional optimization (other dimension is time)
            Star.plot_vecs.append(np.array([1.], dtype=float))
            Star.plot_vecs.append(np.array([-1.], dtype=float))
        else:
            assert plot_settings.num_angles >= 3, "needed at least 3 directions in plot_settings.num_angles"

            step = 2.0 * math.pi / plot_settings.num_angles

            for theta in np.arange(0.0, 2.0*math.pi, step):
                x = math.cos(theta)
                y = math.sin(theta)

                vec = np.array([x, y], dtype=float)

                Star.plot_vecs.append(vec)

    def add_time_to_1d_plot_pts(self, pts):
        '''add time values for plotting 1-d outputs over time

        returns new points list
        '''

        assert len(pts) == 1 or len(pts) == 2, "pts len was {} != 1 or 2 in 1d-plot".format(len(pts))

        if len(pts) == 1: # min == max
            pts.append(pts[0])

        cur_time = self.time_elapse.next_step * self.settings.step
        half_step = self.settings.step / 2.0
        t1 = cur_time - half_step
        t2 = cur_time + half_step

        if self.settings.plot.xdim_dir is None:
            y1 = pts[0][0]
            y2 = pts[1][0]

            pts = [(t1, y1), (t1, y2), (t2, y2), (t2, y1)]
        elif self.settings.plot.ydim_dir is None:
            x1 = pts[0][0]
            x2 = pts[1][0]

            pts = [(x1, t1), (x2, t1), (x2, t2), (x1, t2)]

        return pts

    def verts(self):
        'get the verticies of the polygon projection of the star used for plotting'

        assert Star.plot_settings is not None, "init_plot_vecs() should be called before verts()"

        if self._verts is None:
            use_binary_search = True

            if Star.high_vert_mode:
                use_binary_search = False

            pts = self._find_star_boundaries(use_binary_search=use_binary_search)

            if self.settings.plot.xdim_dir is None or self.settings.plot.ydim_dir is None:
                pts = self.add_time_to_1d_plot_pts(pts)

            if len(pts) > len(Star.plot_vecs)/2 and not Star.high_vert_mode:
                # don't use binary search anymore, and reduce the number of directions being plotted

                Star.high_vert_mode = True
                new_vecs = []

                if len(Star.plot_vecs) > 32:
                    for i in xrange(len(Star.plot_vecs)):
                        if i % 4 == 0:
                            new_vecs.append(Star.plot_vecs[i])

                    Star.plot_vecs = new_vecs

            verts = [[pt[0], pt[1]] for pt in pts]

            # wrap polygon back to first point
            verts.append(verts[0])

            self._verts = verts

        return self._verts

    def _binary_search_star_boundaries(self, start, end, start_point, end_point):
        '''
        return all the optimized points in the star for the passed-in directions, between
        the start and end indices, exclusive

        points which match start_point or end_point are not returned
        '''

        star_lpi = self.get_plot_lpi()

        dirs = Star.plot_vecs
        rv = []

        if start + 1 < end:
            mid = (start + end) / 2
            mid_point = np.zeros(2)

            star_lpi.minimize(dirs[mid], mid_point, error_if_infeasible=True)

            not_start = not np.array_equal(start_point, mid_point)
            not_end = not np.array_equal(end_point, mid_point)

            if not_start:
                rv += self._binary_search_star_boundaries(start, mid, start_point, mid_point)

            if not_start and not_end:
                rv.append(mid_point)

            if not np.array_equal(end_point, mid_point):
                rv += self._binary_search_star_boundaries(mid, end, mid_point, end_point)

        return rv

    def _find_star_boundaries(self, use_binary_search=True):
        '''
        find a constaint-star's boundaries using Star.plot_vecs. This solves several LPs and
        returns a list of points on the boundary (in the standard basis) which maximize each
        of the passed-in directions
        '''

        star_lpi = self.get_plot_lpi()

        point = np.zeros(self.num_plot_vars)
        direction_list = Star.plot_vecs
        rv = []

        assert len(direction_list) >= 2

        if not use_binary_search or len(direction_list) < 8:
            # straightforward approach: minimize in each direction
            last_point = None

            for direction in direction_list:
                star_lpi.minimize(direction, point, error_if_infeasible=True)

                if last_point is None or not np.array_equal(point, last_point):
                    last_point = point.copy()
                    rv.append(last_point)

        else:
            # optimized approach: do binary search to find changes
            star_lpi.minimize(direction_list[0], point, error_if_infeasible=True)
            rv.append(point.copy())

            # add it in thirds, to ensure we don't miss anything
            third = len(direction_list) / 3

            # 0 to 1/3
            star_lpi.minimize(direction_list[third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(0, third, rv[-1], point)
                rv.append(point.copy())

            # 1/3 to 2/3
            star_lpi.minimize(direction_list[2*third], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(third, 2*third, rv[-1], point)
                rv.append(point.copy())

            # 2/3 to end
            star_lpi.minimize(direction_list[-1], point, error_if_infeasible=True)

            if not np.array_equal(point, rv[-1]):
                rv += self._binary_search_star_boundaries(2*third, len(direction_list) - 1, rv[-1], point)
                rv.append(point.copy())

        # pop last point if it's the same as the first point
        if len(rv) > 1 and np.array_equal(rv[0], rv[-1]):
            rv.pop()

        return rv
