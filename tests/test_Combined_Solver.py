"""
I run all these checks in two dimensions before hitting a pull request.
TODO: add checks in higher dimensions or make sure that these checks are
representative of higher dimensions
"""
import numpy as np
import yroots.M_maker as M_maker
from yroots.ChebyshevApproximator import chebApproximate
from yroots.ChebyshevSubdivisionSolver import solveChebyshevSubdivision
import pytest
import unittest
from yroots.polynomial import MultiCheb, MultiPower
from yroots.utils import transform
from yroots.Combined_Solver import solve
import inspect
import sympy as sy

def compare_solve_mmaker(funcs, a, b):
    """
    For checking roots found by solve() against the roots found with M_maker approximator (what previously used 
    for approximations in the code before changed to using approximator in ChebyshevApproximator.py).
    Does so by ensuring that the roots found by solve() give the same result as the roots found by plugging the M_maker 
    approximation outputs into solveChebyshevSubdivision() from ChebyshevSubdivisionSolver.py.

    Parameters
    ----------
    funcs: list 
           collection of callable functions on [-1,1]^n (list w/ funcs for system of eq. want find common roots for!) 
    a : ndarray
        lower interval bounds for interval searching over roots for
    b : ndarray
        upper interval bounds for interval searching over roots for

    Returns
    -------
    2 cases: both cases return a bool for whether or not solve() accomplishes the same results as plugging 
    the output of M_maker approximations into solveChebyshevSubdivision().
    
    (a) 1st case: no roots are found
    bool : ensures that both solve() and M_maker approx. method return no roots
    (b) 2nd case: roots are found
    bool : whether or not roots found by solve() are close enough to roots found by M_maker method to be considered the same
    """
    f, g = funcs  # contains the funcs for the system of eq. want find common roots for

    # get roots found by solve()
    yroots = solve(funcs, a, b)

    # want find approximations on interval [-1,1] so get those lower, upper arrays here
    # also use these instead of a, b here in case a,b were not inputted in [-1,1] form
    lower_itl = -np.ones(len(a))
    upper_itl = np.ones(len(b))

    # get approximations from M_maker file (previous approximator used in code)
    f_approx = M_maker.M_maker(f, lower_itl, upper_itl, f_deg)
    g_approx = M_maker.M_maker(g, lower_itl, upper_itl, g_deg)

    # plug M_maker approx. into solveChebyshevSubdivision to get roots of those approximations
    M_maker_roots = np.array(solveChebyshevSubdivision([f_approx.M, g_approx.M], np.array([f_approx.err, g_approx.err])))

    # use transform from yroots.utils to transform pts. from [-1,1] that used for approx. to interval [a,b] that was inputted
    if len(M_maker_roots) > 0:    # transform only works on non empty arrays
        M_maker_roots = transform(M_maker_roots, a, b)
    else:                         # case where no roots are found
        return len(yroots) == 0   # case (a) for what is returned in this function 

    # check if yroots and roots found from M_maker approx are close enough to equal w/ np.allclose
    solve_matches_M_maker = np.allclose(yroots, M_maker_roots)   # case (b) for what is returned in this function

    return solve_matches_M_maker

def compare_solve_chebapprox(funcs, a, b):
    """
    For checking roots found by solve() against the roots found with ChebyshevApproximator approximator (current method for
    finding approximations in the code).
    Does so by ensuring that the roots found by solve() give the same result as the roots found by plugging the ChebyshevApproximator
    approximation outputs into solveChebyshevSubdivision() from ChebyshevSubdivisionSolver.py.

    Parameters
    ----------
    funcs: list 
           collection of callable functions on [-1,1]^n (list w/ funcs for system of eq. want find common roots for!) 
    a : ndarray
        lower interval bounds for interval searching over roots for
    b : ndarray
        upper interval bounds for interval searching over roots for

    Returns
    -------
    2 cases: both cases return a bool for whether or not solve() accomplishes the same results as plugging 
    the output of ChebyshevApproximator approximations into solveChebyshevSubdivision().
    
    (a) 1st case: no roots are found
    bool : ensures that both solve() and ChebyshevApproximator method return no roots
    (b) 2nd case: roots are found
    bool : whether or not roots found by solve() are close enough to roots found by ChebyshevApproximator method 
    to be considered the same
    """

    f, g = funcs  # contains the funcs for the system of eq. want find common roots for

    # get roots found by solve()
    yroots = solve(funcs, a, b)

    # want find approximations on interval [-1,1] so get those lower, upper arrays here
    # also use these instead of a, b here in case a,b were not inputted in [-1,1] form
    lower_itl = -np.ones(len(a))
    upper_itl = np.ones(len(b))
    
    # get the approximations from ChebyshevApproximator (new approximator being used in code)
    chebapprox_f_approx, chebapprox_f_err = chebApproximate(f, lower_itl, upper_itl)
    chebapprox_g_approx, chebapprox_g_err = chebApproximate(g, lower_itl, upper_itl)

    # plug new approx. into solveChebyshevSubdivision to get roots of those approx.
    cheb_approx_roots = np.array(solveChebyshevSubdivision([chebapprox_f_approx, chebapprox_g_approx],
                                                          np.array([chebapprox_f_err, chebapprox_g_err])))

    # since found approx on [-1,1] transform onto [a,b] interval inputted now
    if len(cheb_approx_roots) > 0:
        cheb_approx_roots = transform(cheb_approx_roots, a, b)
    else:
        return len(yroots) == 0

    # check if roots found from new approx are close enough to yroots to be considered same
    solve_matches_cheb_approx = np.allclose(yroots, cheb_approx_roots)

    return solve_matches_cheb_approx
    
def compare_mmaker_chebapprox(funcs, a, b):
    """
    Tests the currently used approximation method in the code (ChebyshevApproximator.py) against the previous method
    used for finding approximations (M_maker).
    Does so by ensuring that the roots found by plugging both the M_maker and ChebyshevApproximator approximation outputs
    into solveChebyshevSubdivision() in ChebyshevSubidivisionSolver.py are close enough to eachother to be considered the same.
    
    Parameters
    ----------
    funcs: list 
           collection of callable functions on [-1,1]^n (list w/ funcs for system of eq. want find common roots for!) 
    a : ndarray
        lower interval bounds for interval searching over roots for
    b : ndarray
        upper interval bounds for interval searching over roots for

    Returns
    -------
    bool : whether or not M_maker and ChebyshevApproximator approximations accomplish the same result as one another
    """
    
    f, g = funcs  # contains the funcs for the system of eq. want find common roots for

    # get roots found by solve()
    yroots = solve(funcs, a, b)

    # want find approximations on interval [-1,1] so get those lower, upper arrays here
    # also use these instead of a, b here in case a,b were not inputted in [-1,1] form
    lower_itl = -np.ones(len(a))
    upper_itl = np.ones(len(b))

    # get approximations from M_maker file (previous approximator used in code)
    M_f_approx = M_maker.M_maker(f, lower_itl, upper_itl, f_deg)
    M_g_approx = M_maker.M_maker(g, lower_itl, upper_itl, g_deg)

    # plug M_maker approx. into solveChebyshevSubdivision to get roots of those approximations
    M_maker_roots = np.array(solveChebyshevSubdivision([M_f_approx.M, M_g_approx.M], np.array([M_f_approx.err, M_g_approx.err])))

    # use transform from yroots.utils to transform pts. from [-1,1] that used for approx. to interval [a,b] that was inputted
    if len(M_maker_roots) > 0:    # transform only works on non empty arrays
        M_maker_roots = transform(M_maker_roots, a, b)

    # get the approximations from ChebyshevApproximator (new approximator being used in code)
    chebapprox_f_approx, chebapprox_f_err = chebApproximate(f, lower_itl, upper_itl)
    chebapprox_g_approx, chebapprox_g_err = chebApproximate(g, lower_itl, upper_itl)

    # plug new approx. into solveChebyshevSubdivision to get roots of those approx.
    cheb_approx_roots = np.array(solveChebyshevSubdivision([chebapprox_f_approx, chebapprox_g_approx],
                                                          np.array([chebapprox_f_err, chebapprox_g_err])))

    # since found approx on [-1,1] transform onto [a,b] interval inputted now
    if len(cheb_approx_roots) > 0:
        cheb_approx_roots = transform(cheb_approx_roots, a, b)

    # check if roots found from new approx are close enough to yroots to be considered same
    mmaker_matches_cheb_approx = np.allclose(M_maker_roots, cheb_approx_roots)

    return mmaker_matches_cheb_approx

def test_solver():
    """
    runs solver_check() on the six following cases:
    (a) non-[-1,1]^n region of the space
        (i) non-MultiCheb objects
        (ii) some MultiCheb, some non-MultiCheb objects
        (iii) MultiCheb objects
    (b) ^same as above, but on [-1,1]^n region of the space
    """
    a = -1*np.random.random(2)
    b = np.random.random(2)
    arr_neg1 = -np.ones(len(a))
    arr_1 = np.ones(len(b))

    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
    h = MultiCheb(g_approx.M)
    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    k = MultiCheb(f_approx.M)

    #assert compare_solve_mmaker([f,g],a,b) == True   #none multicheb and not neg1_1
    #assert compare_solve_mmaker([f,h],a,b) == True   #some multicheb and not neg1_1
    #assert compare_solve_mmaker([h,k],a,b) == True   #all multicheb and not neg1_1
    b = np.ones(2).astype(float)
    a = -1*b
    assert compare_solve_mmaker([f,g],a,b) == True   #none multicheb and neg1_1
    assert compare_solve_mmaker([k,g],a,b) == True   #some multicheb and neg1_1
    #assert compare_solve_mmaker([h,k],a,b) == True   #all multicheb and neg1_1
    
    return True

def test_invalid_intervals_fail():
    """
    Tests rejection of invalid intervals by solve() to ensure inputted intervals
    a,b are valid for the problem.

    Test Cases: solve() raises a ValueError in these 2 cases
    (a) upper and lower bounding arrays are unequal in length, they have a mismatch in
    dimensions btw. their respective sizes or lengths
    (b) at least one lower bound is greater than or equal to an upper bound

    Returns:
    - True if all test cases pass successfully (if solve() successfully raises a ValueError
    when invalid intervals are inputted)

    Raises:
    - ValueError: If any of the test cases fail (if solve() fails to register an invalid interval), 
    a ValueError is raised with a specific error message indicating the reason for the failure.
    """
    # test case (a)
    # cover cases when a,b have diff num elements along same dim
    # a,b both 1D but a has more elements than b
    a,b = np.array([1,1,1]), np.array([1,1])  # solve() checks case (a) before (b) so not a prob that a>=b here
    with pytest.raises(ValueError) as excinfo:
        solve([f,g], a, b, [f_deg, g_deg])
    assert excinfo.value.args[0] == "Dimension mismatch in intervals."

    # a,b both 1D but a has less elements than b
    # this case also covers that intervals given as lists are correctly converted to np.arrays - yahoo!
    a = [a[0]]
    with pytest.raises(ValueError) as excinfo:
        solve([f,g],a,b,[f_deg,g_deg])
    assert excinfo.value.args[0] == "Dimension mismatch in intervals."

    # non 1D case to be more complicated for funsies
    a = np.array([[1,1],
                  [1,1]])
    b = np.array([[[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]])
    with pytest.raises(ValueError) as excinfo:
        solve([f,g],a,b,[f_deg,g_deg])
    assert excinfo.value.args[0] == "Dimension mismatch in intervals."

    # test case (b)
    a,b = np.array([1,-1]), np.array([1,1])
    with pytest.raises(ValueError) as excinfo:
        solve([f,g],a,b,[f_deg,g_deg])
    assert excinfo.value.args[0] == "At least one lower bound is >= an upper bound."
    
    return True

def test_exact_option():
    """
    Solve has an "exact" option. 
    This tests that option on test case 2.3 from chebfun2_suite.
    We find the roots using the exact method and non-exact method.
    Then we make sure we got the same roots between the two, and that those roots are correct.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]
    yroots_non_exact = solve(funcs,a,b,guess_degs,exact=False)
    yroots_exact = solve(funcs,a,b,guess_degs,exact=True)

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    assert len(yroots_non_exact) == len(actual_roots)
    assert len(yroots_exact) == len(actual_roots)
    assert len(yroots_exact) == len(chebfun_roots)

    actual_roots = ChebyshevSubdivisionSolver.sortRoots(actual_roots)
    yroots_non_exact = ChebyshevSubdivisionSolver.sortRoots(yroots_non_exact)
    yroots_exact = ChebyshevSubdivisionSolver.sortRoots(yroots_exact) 
    chebfun_roots = ChebyshevSubdivisionSolver.sortRoots(chebfun_roots) #sort the Roots

    assert np.allclose(yroots_exact,actual_roots)
    assert np.allclose(yroots_exact,chebfun_roots)
    assert np.allclose(yroots_non_exact,actual_roots)
    assert np.allclose(yroots_non_exact,chebfun_roots)
    
    return true

def testreturnBoundingBoxes():
    """
    Solve has an option to return the bounding boxes on the roots. 
    This test makes sure each root lies within their respective box.
    This uses test case "" from chebfun2_suite
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]

    yroots, boxes = solve(funcs,a,b,guess_degs,returnBoundingBoxes=True)

    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True

    return true
    
def testoutside_neg1_pos1():
    """
    Let the search interval be larger than [-1,1]^n.
    Assert that each root is in its respective box.
    This uses test case "" from chebfun2_suite
    """
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    a,b = np.array([-2,-2]), np.array([2,2])
    funcs = [f,g]
    f_deg,g_deg = 16,16
    guess_degs = [f_deg,g_deg]
    
    yroots, boxes = solve(funcs,a,b,guess_degs,returnBoundingBoxes=True)
    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True
    
    return true

def test_default_nodeg():
    """
    Checks that the solver gets the correct solver when no guess degree is specified.
    Using test case "" from chebfun2_suite.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]

    yroots = solve(funcs,a,b)

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    actual_roots = ChebyshevSubdivisionSolver.sortRoots(actual_roots)
    chebfun_roots = ChebyshevSubdivisionSolver.sortRoots(chebfun_roots) #sort the Roots
    yroots = ChebyshevSubdivisionSolver.sortRoots(yroots) 

    assert np.allclose(yroots,actual_roots)
    assert np.allclose(yroots,chebfun_roots)
    
    return true

def test_deg_inf():
    """
    Tests the logic in Combined_Solver.py that detects which functions are MultiCheb, non-MultiCheb
    and which functions can be treated like polynomials. This information is used to make smart degree
    guesses, and the logic used to make the guesses is tested as well.
    """
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    h = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]), np.array([1,1])
    g_deg = 3
    g = MultiCheb(M_maker.M_maker(g,a,b,g_deg).M)
    funcs = [f,g,h]
    guess_degs = None

    default_deg = 2
    is_lambda_poly, is_routine, is_lambda, guess_degs = degree_guesser(funcs,guess_degs,default_deg)

    assert (is_lambda_poly == np.array([True, False, False])).all()
    assert (is_routine == np.array([True,False,True])).all()
    assert (is_lambda == np.array([True,False,True])).all() #TODO:need a test case for python functions with lambda not in the function definition, so is_routine is not is_lambda
    assert (guess_degs == np.array([3,3,2])).all()
    
    return True
    
if __name__ == '__main__':
    f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    f_deg,g_deg = 20,20
    a, b = -np.ones(2), np.ones(2)
    
    #Test each function now by actually calling them!
    
    tests_passed = 0     #Will act as counter for printing out if all tests were passed at the end

    solve_matches_mmaker = compare_solve_mmaker([f,g],a,b)
    solve_matches_cheb_approx = compare_solve_chebapprox([f,g],a,b)
    mmaker_matches_chebapprox = compare_mmaker_chebapprox([f,g],a,b)

    if (solve_matches_mmaker):
        tests_passed += 1
    else:
        print("Solve() function failed to get same result as M_maker approximator when finding roots")

    if (solve_matches_cheb_approx):
        tests_passed += 1
    else:
        print("Solve() function failed to get same result as chebApproximate() when finding roots")

    if (mmaker_matches_chebapprox):
        tests_passed += 1
    else:
        print("The new chebApproximate gives the same result as the previous M_maker approximator when finding roots")
    
    """if (test_solver()):
        tests_passed += 1
    if (test_invalid_intervals_fail()):
        tests_passed += 1"""
    if (test_exact_option()):
        tests_passed += 1
    """if (testreturnBoundingBoxes):
        tests_passed += 1
    if (testoutside_neg1_pos1()):
        tests_passed += 1
    if (test_default_nodeg()):
        tests_passed += 1
    if (test_deg_inf()):
        tests_passed += 1"""
    
    #Print out message saying if all tests were passed
    if (tests_passed == 10):
        print("SUCCESS!!! ALL TESTS PASSED!!!") 
pass
