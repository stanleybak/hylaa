'''
Hylaa Symoblic dynamics construction

Construct A matrix and reset matrix / rhs from symbolic expressions

Stanley Bak
Nov 2018
'''

import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import Mul, Expr, Add, Symbol, Number

def extract_linear_terms(e, variables, has_affine_variable):
    '''extract linear terms from a flat sympy expression

    returns a list of numbers, one for each variable
    '''

    rv = [0] * len(variables)

    if has_affine_variable:
        rv.append(0)

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    try:
        _extract_linear_terms_rec(e, variables, rv, has_affine_variable)
    except RuntimeError as ex:
        raise RuntimeError(str(ex) + ", while parsing " + str(e))

    return rv

def _extract_linear_terms_rec(e, variables, rv, has_affine_variable):
    'extract linear terms'

    if isinstance(e, Add):
        _extract_linear_terms_rec(e.args[0], variables, rv, has_affine_variable)

        for arg in e.args[1:]:
            _extract_linear_terms_rec(arg, variables, rv, has_affine_variable)
    elif isinstance(e, Number):
        val = float(e)

        if val != 0:
            if not has_affine_variable:
                raise RuntimeError(f"expression has affine variables but has_affine_variable was False: '{e}'")

            rv[-1] += val
    elif isinstance(e, Symbol):
        try:
            index = variables.index(e.name)
        except ValueError:
            raise RuntimeError(f"variable {e.name} not found in variable list: {variables}")

        rv[index] += 1      
    elif isinstance(e, Mul):
        if len(e.args) != 2:
            raise RuntimeError(f"expected multiplication term with exactly two arguments: '{e}'")

        sym_term = None
        num_term = None

        if isinstance(e.args[0], Number) and isinstance(e.args[1], Symbol):
            num_term = e.args[0]
            sym_term = e.args[1]
        elif isinstance(e.args[1], Number) and isinstance(e.args[0], Symbol):
            num_term = e.args[1]
            sym_term = e.args[0]
        else:
            raise RuntimeError(f"expected multiplication with one number and one variable: '{e}'")

        try:
            index = variables.index(sym_term.name)
        except ValueError:
            raise RuntimeError(f"variable {sym_term.name} not found in variable list: {variables}")

        rv[index] += float(num_term)
    else:
        raise RuntimeError(f"unsupported term of type {type(e)}: '{e}'")

def make_reset_mat(variables, resets, constant_dict, has_affine_variable=False):
    'make the matrix for a reset operation'

    mat = make_dynamics_mat(variables, resets, constant_dict, has_affine_variable=has_affine_variable)

    if has_affine_variable:
        # affine variable should be identity reset, rather than all 0's as in dynamics
        mat[-1][-1] = 1.0

    return mat

def make_dynamics_mat(variables, derivatives, constant_dict, has_affine_variable=False):
    '''make the dynamics A matrix from the list of variables, derivatives, and a dict mapping constants to values

    returns a list of lists (a matrix) of size len(variables) by len(variables)
    '''

    rv = []
    subs = {}
    symbol_dict = {}
    
    for var in variables:
        symbol_dict[var] = sympy.symbols(var)
    
    for var, value in constant_dict.items():
        sym_var = sympy.symbols(var)
        subs[sym_var] = value
        symbol_dict[var] = sym_var

    for der in derivatives:
        sym_der = parse_expr(der, local_dict=symbol_dict)

        sym_der = sym_der.subs(subs)

        row = extract_linear_terms(sym_der, variables, has_affine_variable)
        rv.append(row)

    if has_affine_variable:
        rv.append([0] * (len(variables) + 1))

    return rv

def make_condition(variables, condition_list, constant_dict, has_affine_variable=False):
    '''make a condition matrix and right-hand-side (rhs) from a set of string conditions.
    condition_list is a list of strings with a single '<=' or '>=' condition like 'x - 1 + y <= 2 * x + 3'

    returns a 2-tuple: (mat, rhs)
    '''

    assert isinstance(condition_list, list), "condition_list should be a list of string conditions"

    mat = []
    rhs = []
    subs = {}
    symbol_dict = {}

    for var in variables:
        symbol_dict[var] = sympy.symbols(var)
    
    for var, value in constant_dict.items():
        sym_var = sympy.symbols(var)
        subs[sym_var] = value
        symbol_dict[var] = sym_var

    for cond in condition_list:
        less_than_count = cond.count('<=')
        greater_than_count = cond.count('>=')

        if less_than_count + greater_than_count != 1:
            raise RuntimeError(f"Expected condition with single '<=' or '>=': {cond}")

        if greater_than_count == 1:
            cond = cond.replace(">=", "<=")

        left, right = cond.split("<=")

        # swap left and right for '>=' expressions
        if greater_than_count == 1:
            left, right = right, left

        # make the expression: left - (right) <= 0
        subtract_cond = f"{left} - ({right})"
        
        sym_cond = parse_expr(subtract_cond, local_dict=symbol_dict)

        # substitute in constants
        sym_cond = sym_cond.subs(subs)

        terms = extract_linear_terms(sym_cond, variables, has_affine_variable=True)
        row = terms[:-1]

        if has_affine_variable:
            row.append(0)
        
        mat.append(row)
        rhs.append(-1 * terms[-1])

    return mat, rhs
