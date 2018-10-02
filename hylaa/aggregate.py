'''
Stanley Bak

Implementation of concrete aggregation methods, used by aggdag
'''

import numpy as np

from hylaa.stateset import StateSet
from hylaa import lputil

def aggregate_box_arnoldi(agg_list, op_list, is_box, is_arnoldi, add_guard, print_func):
    '''
    perform template-based aggregation on the passed-in list of states

    Currently, this can either use box template directions or arnoldi (+box) template directions
    '''

    assert is_box or is_arnoldi

    min_step = min([state.cur_steps_since_start[0] for state in agg_list])
    max_step = max([state.cur_steps_since_start[1] for state in agg_list])
    step_interval = [min_step, max_step]

    print_func("Aggregation time step interval: {}".format(step_interval))

    # create a new state from the aggregation
    postmode = agg_list[0].mode
    postmode_dims = postmode.a_csr.shape[0]
    mid_index = len(agg_list) // 2

    op = op_list[mid_index]

    if is_box or op is None:
        agg_dir_mat = np.identity(postmode_dims)
    elif is_arnoldi:
        # aggregation with a predecessor, use arnoldi directions in predecessor mode in center of
        # middle aggregagted state, then project using the reset, and reorthogonalize

        premode = op.parent_node.stateset.mode
        t = op.transition
        print_func("aggregation point: {}".format(op.premode_center))

        premode_dir_mat = lputil.make_direction_matrix(op.premode_center, premode.a_csr)
        print_func("premode dir mat:\n{}".format(premode_dir_mat))

        if t.reset_csr is None:
            agg_dir_mat = premode_dir_mat
        else:
            projected_dir_mat = premode_dir_mat * t.reset_csr.transpose()

            print_func("projected dir mat:\n{}".format(projected_dir_mat))

            # re-orthgohonalize (and create new vectors if necessary)
            agg_dir_mat = lputil.reorthogonalize_matrix(projected_dir_mat, postmode_dims)

        # also add box directions in target mode (if they don't already exist)
        box_dirs = []
        for dim in range(postmode_dims):
            direction = [0 if d != dim else 1 for d in range(postmode_dims)]
            exists = False

            for row in agg_dir_mat:
                if np.allclose(direction, row):
                    exists = True
                    break

            if not exists:
                box_dirs.append(direction)

        if box_dirs:
            agg_dir_mat = np.concatenate((agg_dir_mat, box_dirs), axis=0)

    if op and add_guard:
        # add all the guard conditions to the agg_dir_mat
        
        t = op.transition

        if t.reset_csr is None: # identity reset
            guard_dir_mat = t.guard_csr
        else:
            # multiply each direction in the guard by the reset
            guard_dir_mat = t.guard_csr * t.reset_csr.transpose()

        if guard_dir_mat.shape[0] > 0:
            agg_dir_mat = np.concatenate((agg_dir_mat, guard_dir_mat.toarray()), axis=0)

    print_func("agg dir mat:\n{}".format(agg_dir_mat))
    lpi_list = [state.lpi for state in agg_list]

    new_lpi = lputil.aggregate(lpi_list, agg_dir_mat, postmode)

    return StateSet(new_lpi, agg_list[0].mode, step_interval, op_list, is_concrete=False)

def aggregate_chull(agg_list, op_list, print_func):
    '''
    perform template-based aggregation on the passed-in list of states

    Currently, this can either use box template directions or arnoldi (+box) template directions
    '''

    min_step = min([state.cur_steps_since_start[0] for state in agg_list])
    max_step = max([state.cur_steps_since_start[1] for state in agg_list])
    step_interval = [min_step, max_step]

    print_func("Convex hull aggregation time step interval: {}".format(step_interval))

    postmode = agg_list[0].mode
    lpi_list = [state.lpi for state in agg_list]

    new_lpi = lputil.aggregate_chull(lpi_list, postmode)

    return StateSet(new_lpi, agg_list[0].mode, step_interval, op_list, is_concrete=False)
