'''
Utilities for testing

Stanley Bak, 2018
'''

def pair_almost_in(pair, pair_list, tol=1e-9):
    'check if a pair is in a pair list (up to small tolerance)'

    rv = False

    for a, b in pair_list:
        if abs(a - pair[0]) < tol and abs(b - pair[1]) < tol:
            rv = True
            break

    return rv

def assert_verts_is_box(verts, box, tol=1e-5):
    '''check that a list of verts is almost equal to the passed-in box using assertions

    box is [[xmin, xmax], [ymin, ymax]]
    '''

    is_point = abs(box[0][0] - box[0][1]) < tol and abs(box[1][0] - box[1][1]) < tol
    is_flat = abs(box[0][0] - box[0][1]) < tol or abs(box[1][0] - box[1][1]) < tol

    expected_verts = 2 if is_point else 3 if is_flat else 5

    assert len(verts) == expected_verts and verts[0] == verts[-1]

    pts = [(box[0][0], box[1][0]), (box[0][1], box[1][0]), (box[0][1], box[1][1]), (box[0][0], box[1][1])]

    for pt in pts:
        found = False

        for vert in verts:
            x, y = vert

            if abs(x - pt[0]) < tol and abs(y - pt[1]) < tol:
                found = True
                break

        assert found, "Point {} was not found in verts: {}".format(pt, verts)
