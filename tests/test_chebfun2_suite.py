# The purpose of this file is to run the chebfun2_suite tests we expect to pass with our current code with the unit tests so that can check any new pull requests with these tests!
# As of 6/26/23 the following tests should pass perfectly with our current code:
# 1.1, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.2, 4.1, 5.1, 6.2, 6.3, 7.1, 7.3, 8.1, 8.2, 9.1, 9.2
# Any tests from the chebfun suite that are not any of these ^ have been removed for testing purposes bc with the unit tests, we just want to make sure they pass the ones we expect them to!

import numpy as np
from yroots.Combined_Solver import solve
from time import time
from matplotlib import pyplot as plt
from yroots.utils import sortRoots
import unittest

def norm_pass_or_fail(yroots, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not their norms are within tol of the norms of the
        "actual" roots, which are determined either by previously known
        roots or Marching Squares roots.
    Parameters
    ----------
        yroots : numpy array
            The roots that yroots found.
        roots : numpy array
            "Actual" roots obtained via mp.findroot
        tol : float, optional
            Tolerance that determines how close the roots need to be in order
            to be considered close. Defaults to 1000*eps where eps is machine
            epsilon.

    Returns
    -------
         bool
            Whether or not all the roots were close enough.
    """
    roots_sorted = sortRoots(roots)
    yroots_sorted = sortRoots(yroots)
    root_diff = roots_sorted - yroots_sorted
    return np.linalg.norm(root_diff[:,0]) < tol and np.linalg.norm(root_diff[:,1]) < tol


def residuals(func, roots):
    """ Finds the residuals of the given function at the roots.
    Paramters
    ---------
        func : function
            The function to find the residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        numpy array
            The residuals of the function.
    """
    return np.abs(func(roots[:,0],roots[:,1]))


def residuals_pass_or_fail(funcs, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not the maximal residuals are within a certain tolerance.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the residuals of.
        roots : numpy array
            The roots to plug into the functions to get the residuals.
        tol : float, optional
            How close to 0 the maximal residual must be in order to pass.
            Defaults to 1000* eps where eps is machine epsilon.
    Returns
    -------
        bool
            True if the roots pass the test (are close enough to 0), False
            otherwise.
    """
    for func in funcs:
        if np.max(residuals(func, roots)) > tol:
            return False

    return True


def pass_or_fail(funcs, yroots, roots, test_num, test_type="norm", tol=2.220446049250313e-13):
    """Determines whether a test passes or fails bsed on the given criteria.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the roots of.
        yroots : numpy array
            Roots found by yroots.
        roots : numpy array
            The list of "actual" or Marching Squares roots.
        test_num : float or string
            The number of the test. For example, test 9.2 one could pass in
            "9.2" or 9.2.
        test_type : string, optional
            What type of test to use to determine wheter it passes or fails.
             - "norm" -- runs norm_pass_or_fail, default
             - "residual" -- runs residual_pass_or_fail
        tol : float, optional
            The tolerance with which we want to run our tests. Defualts to
            1000*eps where eps is machine epsilon.
    Raises
    ------
        AssertionError
            If len(yroots) != len(roots) or if it fails the residual
            or norm tests.
        ValueError
            If test_type is not "norm" or "residual"
    """
    if (test_type not in ['norm','residual']):
        raise ValueError("test_type must be 'norm' or 'residual'.")

    if len(yroots) != len(roots):
        if len(yroots) > len(roots):
            raise AssertionError("Test " + str(test_num) +  ": YRoots found"
                                 " too many roots: " + str(len(yroots)) +
                                 " where " + str(len(roots)) + " were expected.")
        else:
            raise AssertionError("Test " + str(test_num) +  ": YRoots didn't"
                                 " find enough roots: " + str(len(yroots)) +
                                 " where " + str(len(roots)) + " were expected.")

    if test_type == 'norm':
        assert norm_pass_or_fail(yroots, roots, tol=tol), "Test " + str(test_num) + " failed."
    else:
        assert residuals_pass_or_fail(funcs, yroots, tol=tol), "Test " + str(test_num) + " failed."


def norm_pass_or_fail(yroots, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not their norms are within tol of the norms of the
        "actual" roots, which are determined either by previously known
        roots or Marching Squares roots.
    Parameters
    ----------
        yroots : numpy array
            The roots that yroots found.
        roots : numpy array
            "Actual" roots either obtained analytically or through Marching
            Squares.
        tol : float, optional
            Tolerance that determines how close the roots need to be in order
            to be considered close. Defaults to 1000*eps where eps is machine
            epsilon.

    Returns
    -------
         bool
            Whether or not all the roots were close enough.
    """
    roots_sorted = sortRoots(roots)
    yroots_sorted = sortRoots(yroots)
    root_diff = roots_sorted - yroots_sorted
    return np.linalg.norm(root_diff[:,0]) < tol and np.linalg.norm(root_diff[:,1]) < tol, np.linalg.norm(root_diff[:,0]), np.linalg.norm(root_diff[:,1])


def residuals(func, roots):
    """ Finds the residuals of the given function at the roots.
    Paramters
    ---------
        func : function
            The function to find the residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        numpy array
            The residuals of the function.
    """
    return np.abs(func(roots[:,0],roots[:,1]))


def residuals_pass_or_fail(funcs, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not the maximal residuals are within a certain tolerance.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the residuals of.
        roots : numpy array
            The roots to plug into the functions to get the residuals.
        tol : float, optional
            How close to 0 the maximal residual must be in order to pass.
            Defaults to 1000* eps where eps is machine epsilon.
    Returns
    -------
        bool
            True if the roots pass the test (are close enough to 0), False
            otherwise.
    """
    for func in funcs:
        if np.max(residuals(func, roots)) > tol:
            return False

    return True

def verbose_pass_or_fail(funcs, yroots, polished_roots, test_num, cheb_roots=None, tol=2.220446049250313e-13):
    """ Determines which tests pass and which fail.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the roots of.
        yroots : numpy array
            Roots found by yroots.
        MSroots : numpy array
            The list of "actual" or Marching Squares roots.
        test_num : float or string
            The number of the test. For example, test 9.2 one could pass in
            "9.2" or 9.2.
        cheb_roots : numpy array
            Chebfun roots for extra comparison when MS are available.
        tol : float, optional
            The tolerance with which we want to run our tests. Defualts to
            1000*eps where eps is machine epsilon.
    Raises
    ------
        AssertionError
            If len(yroots) != len(roots) or if it fails the residual
            or norm tests.
    """
    print ("=========================================================")
    print("Test " + str(test_num))
    #Make sure dimensions are right
    if polished_roots.ndim == 1:
        polished_roots = polished_roots[..., np.newaxis].T
    
    #Fail if the number of roots is wrong
    if len(yroots) != len(polished_roots):
        print(f"\t Num Roots Wrong! Found {len(yroots)}, Has {len(polished_roots)}!")
        return False, False

    residuals_pass = residuals_pass_or_fail(funcs, yroots, tol)
    if residuals_pass:
        print("\t Residual test: pass")
    else:
        print("\t Residual test: fail")

    if cheb_roots is not None:
        if residuals_pass_or_fail(funcs, cheb_roots, tol):
            print("\t Chebfun passes residual test")
        else:
            print("\t Chebfun fails residual test")
        try:
            norm_pass, x_norm, y_norm = norm_pass_or_fail(polished_roots, cheb_roots, tol)
            if norm_pass:
                print("\t Chebfun norm test: pass")
            else:
                print("\t Chebfun norm test: fail")
                print("The norm of the difference in x values:", x_norm)
                print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
            print("A different number of roots were found.")
            print ("Yroots: " + str(len(yroots)))
            print("Chebfun Roots: " + str(len(cheb_roots)))
    if polished_roots is not None:
        try:
            norm_pass, x_norm, y_norm = norm_pass_or_fail(yroots, polished_roots, tol)
            if norm_pass:
                print("\t YRoots norm test: pass")
            else:
                print("\t YRoots norm test: fail")
                print("The norm of the difference in x values:", x_norm)
                print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
                print("A different number of roots were found.")
                print ("Yroots: " + str(len(yroots)))
                print("Polished: " + str(len(polished_roots)))        
    print("")
    print("YRoots max residuals:")
    YR_resid = list()
    for i, func in enumerate(funcs):
        YR_resid.append(residuals(func, yroots))
        print("\tf" + str(i) + ": " + str(np.max(residuals(func, yroots))))

    cheb_resid = None
    if cheb_roots is not None:
        cheb_resid = list()
        print("Chebfun max residuals:")
        for i, func in enumerate(funcs):
            cheb_resid.append(residuals(func, cheb_roots))
            print("\tf" + str(i) + ": " + str(np.max(residuals(func, cheb_roots))))
    if polished_roots is not None:
        print("Polished max residuals:")
        Other_resid = list()
        for i, func in enumerate(funcs):
            Other_resid.append(residuals(func, polished_roots))
            print("\tf" + str(i) + ": " + str(np.max(residuals(func, polished_roots))))

        if len(yroots) > len(polished_roots):
            print("YRoots found more roots.")
            print("=========================================================")
            return residuals_pass,norm_pass

    # print("Comparison of Residuals (YRoots <= Other)")
    num_smaller = 0
    if polished_roots is not None:
        for i in range(len(YR_resid)):
            comparison_array = (YR_resid[i] <= Other_resid[i])
            # print(comparison_array)
            num_smaller += np.sum(comparison_array)
        #print("Number of YRoots residual values <= Polished residual values are: " + str(num_smaller))

    if cheb_resid is not None:
        if len(yroots) > len(cheb_roots):
            print("=========================================================")
            return residuals_pass,norm_pass

        for i in range(len(YR_resid)):
            comparison_array2 = (YR_resid[i] <= cheb_resid[i])
            num_smaller += np.sum(comparison_array2)
    #print("Number of YRoots residual values <= to Chebfun residual values are: " + str(num_smaller))

    print("=========================================================")
    return residuals_pass,norm_pass

#TODO: include test cases for for returnbounding boxes, exact, rescale

def test_roots_1_1():
    # Test 1.1
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g = lambda x,y: y-x**6
    funcs = [f,g]
    a, b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.1, cheb_roots=chebfun_roots)

def test_roots_1_3():
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.3, cheb_roots=chebfun_roots)
    
def test_roots_1_4():
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    actual_roots = np.load('Polished_results/polished_1.4.npy')
    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.4.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.4, cheb_roots=chebfun_roots)

def test_roots_1_5():
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    actual_roots = np.load('Polished_results/polished_1.5.npy')
    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.5.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.5, cheb_roots=chebfun_roots)

def test_roots_2_1():
    # Test 2.1
    f = lambda x,y: np.cos(10*x*y)
    g = lambda x,y: x + y**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.1, cheb_roots=chebfun_roots)

def test_roots_2_2():
    # Test 2.2
    f = lambda x,y: x
    g = lambda x,y: (x-.9999)**2 + y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.2.csv', delimiter=',')
    
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.2, cheb_roots=chebfun_roots)

def test_roots_2_3():
    # Test 2.3
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.3, cheb_roots=chebfun_roots)

def test_roots_2_4():
    # Test 2.4
    f = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    g = lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.4.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.4.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.4, cheb_roots=chebfun_roots)

def test_roots_2_5():
    # Test 2.5
    f = lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    funcs = [f,g]
    a,b = np.array([-4,-4]), np.array([4,4])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.5.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.5.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.5, cheb_roots=chebfun_roots, tol=2.220446049250313e-12)

def test_roots_3_1():
    # Test 3.1
    f = lambda x,y: ((x-.3)**2+2*(y+0.3)**2-1)
    g = lambda x,y: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 3.1, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)

def test_roots_3_2():
    # Test 3.2
    f = lambda x,y: ((x-0.1)**2+2*(y-0.1)**2-1)*((x+0.3)**2+2*(y-0.2)**2-1)*((x-0.3)**2+2*(y+0.15)**2-1)*((x-0.13)**2+2*(y+0.15)**2-1)
    g = lambda x,y: (2*(x+0.1)**2+(y+0.1)**2-1)*(2*(x+0.1)**2+(y-0.1)**2-1)*(2*(x-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 3.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)
    
def test_roots_4_1():
    # Test 4.1
    # This system hs 4 true roots, but ms fails (finds 5).
    f = lambda x,y: np.sin(3*(x+y))
    g = lambda x,y: np.sin(3*(x-y))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_4.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_4.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 4.1, cheb_roots=chebfun_roots)

def test_roots_5():
    # Test 5.1
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-2,-2]), np.array([2,2])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_5.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_5.1.csv', delimiter=',')
    
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 5.1, cheb_roots=chebfun_roots)

def test_roots_6_2():
    # Test 6.2
    f = lambda x,y: (y - 2*x)*(y+.5*x)
    g = lambda x,y: (x-.0001)*(x**2+y**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_6.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_6.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 6.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)

def test_roots_6_3():
    # Test 6.3
    f = lambda x,y: 25*x*y - 12
    g = lambda x,y: x**2+y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_6.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_6.3.csv', delimiter=',')
    
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 6.3, cheb_roots=chebfun_roots)

def test_roots_7_1():
    # Test 7.1
    f = lambda x,y: (x**2+y**2-1)*(x-1.1)
    g = lambda x,y: (25*x*y-12)*(x-1.1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.1, cheb_roots=chebfun_roots)

def test_roots_7_3():
    # Test 7.3
    c = 1.e-09
    f = lambda x,y: np.cos(x*y/(c**2))+np.sin(3*x*y/(c**2))
    g = lambda x,y: np.cos(y/c)-np.cos(2*x*y/(c**2))
    funcs = [f,g]
    a,b = np.array([-1e-9, -1e-9]), np.array([1e-9, 1e-9])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.3.csv', delimiter=',')
    
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.3, cheb_roots=chebfun_roots,tol=2.220446049250313e-10)

def test_roots_8_1():
    # Test 8.1
    f = lambda x,y: np.sin(10*x-y/10)
    g = lambda x,y: np.cos(3*x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_8.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_8.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 8.1, cheb_roots=chebfun_roots)

def test_roots_8_2():
    # Test 8.2
    f = lambda x,y: np.sin(10*x-y/10) + y
    g = lambda x,y: np.cos(10*y-x/10) - x
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_8.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_8.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 8.2, cheb_roots=chebfun_roots)
    
def test_roots_9_1():
    # Test 9.1
    f = lambda x,y: x**2+y**2-.9**2
    g = lambda x,y: np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_9.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_9.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 9.1, cheb_roots=chebfun_roots)

def test_roots_9_2():
    # Test 9.2
    f = lambda x,y: x**2+y**2-.49**2
    g = lambda x,y: (x-.1)*(x*y - .2)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_9.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_9.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 9.2, cheb_roots=chebfun_roots)

if __name__ == "__main__":
    # Run all the tests!
    tests = np.array([test_roots_1_1,
                        test_roots_1_3,
                        test_roots_1_4,
                        test_roots_1_5,
                        test_roots_2_1,
                        test_roots_2_2,
                        test_roots_2_3,
                        test_roots_2_4,
                        test_roots_2_5,
                        test_roots_3_1,
                        test_roots_3_2,
                        test_roots_4_1,
                        test_roots_5,
                        test_roots_6_2,
                        test_roots_6_3,
                        test_roots_7_1,
                        test_roots_7_3,
                        test_roots_8_1,
                        test_roots_8_2,
                        test_roots_9_1,
                        test_roots_9_2])
    res_passes = np.zeros_like(tests,dtype=bool)
    norm_passes = np.zeros_like(tests,dtype=bool)
    for i,test in enumerate(tests):
        t, passes = test()
        res_pass,norm_pass = passes
        res_passes[i] = res_pass
        norm_passes[i] = norm_pass
        
    print('Summary')
    print(f'Residual Test: Passed {np.sum(res_passes)} of 21, {100*np.mean(res_passes)}%')
    where_failed_res = np.where(~res_passes)[0]
    failed_res_tests = tests[where_failed_res]
    assert len(failed_res_tests) == 0, f'Failed Residual Test on \n{[t.__name__ for t in failed_res_tests]}'

    print(f'Norm Test    : Passed {np.sum(norm_passes)} of 21, {100*np.mean(norm_passes)}%')
    where_failed_norm = np.where(~norm_passes)[0]
    failed_norm_tests = tests[where_failed_norm]
    assert len(failed_norm_tests) == 0, f'Failed Norm Test on \n{[t.__name__ for t in failed_norm_tests]}'    

