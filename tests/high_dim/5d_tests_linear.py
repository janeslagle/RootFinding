import numpy as np
import yroots as yr
from yroots.subdivision import solve
from time import time

def residuals(f1,f2,f3,f4,f5,roots,t):
    Resid = list()
    for i in range(len(roots)):
        Resid.append(np.abs(f1(roots[i,0],roots[i,1],roots[i,2],roots[i,3]),roots[i,4]))
        Resid.append(np.abs(f2(roots[i,0],roots[i,1],roots[i,2],roots[i,3]),roots[i,4]))
        Resid.append(np.abs(f3(roots[i,0],roots[i,1],roots[i,2],roots[i,3]),roots[i,4]))
        Resid.append(np.abs(f4(roots[i,0],roots[i,1],roots[i,2],roots[i,3]),roots[i,4]))
        Resid.append(np.abs(f5(roots[i,0],roots[i,1],roots[i,2],roots[i,3]),roots[i,4]))

    hours = int(t // 3600)
    minutes = int((t%3600) // 60)
    seconds = int((t%3600)%60 // 1)
    msecs = int(np.round((t % 1) * 1000,0))
    print("time elapsed: ",hours,"hours,", minutes,"minutes,",seconds, "seconds,",msecs, "milliseconds")
    print("Residuals: ", Resid, "\n")
    print("Max Residual: ", np.amax(Resid))
    return np.amax(Resid)

def ex1_linear():
    f1 = lambda x1,x2,x3,x4,x5: np.sin(x1*x3) + x1*np.log(x2+3) - x1**2
    f2 = lambda x1,x2,x3,x4,x5: np.cos(4*x1*x2) + np.exp(3*x2/(x1-2)) - 5
    f3 = lambda x1,x2,x3,x4,x5: np.cos(2*x2) - 3*x3 + 1/(x1-8)
    f4 = lambda x1,x2,x3,x4,x5: x1 + x2 - x3 - x4
    f5 = lambda x1,x2,x3,x4,x5: x1 + x2 - x3 - x4 + x5


    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f1,f2,f3,f4,f5], a, b,target_deg=1)
    t = time() - start
    print("====================== ex 1 linear ======================")
    return residuals(f1,f2,f3,f4,f5,roots,t)

def ex2_linear():
    f = lambda x,y,z,x4,x5: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z,x4,x5: x - np.log(1/(y+3))
    h = lambda x,y,z,x4,x5: x**2 -  z
    f4 = lambda x,y,z,x4,x5: x + y + z + x4
    f5 = lambda x,y,z,x4,x5: x + y + z + x4 + x5

    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4,f5], a, b,target_deg=1)
    t = time() - start
    print("====================== ex 2 linear ======================")
    return residuals(f,g,h,f4,f5,roots,t)

def ex3_linear():
    f = lambda x,y,z,x4,x5: y**2-x**3
    g = lambda x,y,z,x4,x5: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z,x4,x5: x**2 + y**2 + z**2 - 1
    f4 = lambda x,y,z,x4,x5: x + y + z + x4
    f5 = lambda x,y,z,x4,x5: x + y + z + x4 - x5

    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4,f5], a, b,target_deg=1)
    t = time() - start
    print("====================== ex 3 linear ======================")
    return residuals(f,g,h,f4,f5,roots,t)


if __name__ == "__main__":
    ex2_linear()
    ex3_linear()
