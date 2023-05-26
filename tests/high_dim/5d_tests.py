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

def ex1():
    f1 = lambda x1,x2,x3,x4,x5: np.sin(x1*x3) + x1*np.log(x2+3) - x1**2
    f2 = lambda x1,x2,x3,x4,x5: np.cos(4*x1*x2) + np.exp(3*x2/(x1-2)) - 5
    f3 = lambda x1,x2,x3,x4,x5: np.cos(2*x2) - 3*x3 + 1/(x1-8)
    f4 = lambda x1,x2,x3,x4,x5: x1 + x2 - x3 - x4
    f5 = lambda x1,x2,x3,x4,x5: x1 + x2 - x3 - x4 + x5


    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f1,f2,f3,f4,f5], a, b)
    t = time() - start
    print("====================== ex 1 linear ======================")
    return residuals(f1,f2,f3,f4,f5,roots,t)

def dex1():
    df1 = lambda x1, x2, x3, x4, x5 : (x3*np.cos(x1*x3) + np.log(x2+3) - 2*x1, x1/(x2 + 3), x1*np.cos(x1*x3), 0, 0)
    df2 = lambda x1, x2, x3, x4, x5 : (-4*x2*np.sin(4*x1*x2) - (3*x2/(x1-2)**2)*np.exp(3*x2/(x1-2)), -4*x1*np.sin(4*x1*x2) + 3*np.exp(3*x2/(x1-2))/(x1-2), 0, 0, 0)
    df3 = lambda x1, x2, x3, x4, x5 : (-1/(x1-8)**2, -2*np.sin(2*x2), -3, 0, 0)
    df4 = lambda x1, x2, x3, x4, x5 : (1, 1, -1, -1, 0)
    df5 = lambda x1, x2, x3, x4, x5 : (1, 1, -1, -1, 1)
    return df1, df2, df3, df4, df5

def ex2():
    f = lambda x,y,z,x4,x5: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z,x4,x5: x - np.log(1/(y+3))
    h = lambda x,y,z,x4,x5: x**2 -  z
    f4 = lambda x,y,z,x4,x5: x + y + z + x4
    f5 = lambda x,y,z,x4,x5: x + y + z + x4 + x5

    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4,f5], a, b)
    t = time() - start
    print("====================== ex 2 linear ======================")
    return residuals(f,g,h,f4,f5,roots,t)

def dex2():
    df = lambda x, y, z, x4, x5 : (4*y*np.sinh(4*x*y), 4*x*np.sinh(4*x*y), np.exp(z), 0, 0)
    dg = lambda x, y, z, x4, x5 : (1, 1/(y + 3), 0, 0, 0)
    dh = lambda x, y, z, x4, x5 : (2*x, 0, -1, 0, 0)
    df4 = lambda x,  y, z, x4, x5 : (1, 1, 1, 1, 0)
    df5 = lambda x, y, z, x4, x5 : (1, 1, 1, 1, 1)
    return df, dg, dh, df4, df5

def ex3():
    f = lambda x,y,z,x4,x5: y**2-x**3
    g = lambda x,y,z,x4,x5: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z,x4,x5: x**2 + y**2 + z**2 - 1
    f4 = lambda x,y,z,x4,x5: x + y + z + x4
    f5 = lambda x,y,z,x4,x5: x + y + z + x4 - x5

    a = [-1,-1,-1,-1,-1]
    b = [1,1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4,f5], a, b)
    t = time() - start
    print("====================== ex 3 linear ======================")
    return residuals(f,g,h,f4,f5,roots,t)

def dex3():
    df = lambda x, y, z, x4, x5 : (-3*x**2, 2*y, 0, 0, 0)
    dg = lambda x, y, z, x4, x5 : (-2*(x-.1), 3*(y+.1)**2, 0, 0, 0)
    dh = lambda x, y, z, x4, x5 : (2*x, 2*y, 2*z, 0, 0)
    df4 = lambda x, y, z, x4, x5 : (1, 1, 1, 1, 0)
    df5 = lambda x, y, z, x4, x5 : (1, 1, 1, 1, -1)
    return df, dg, dh, df4, df5

if __name__ == "__main__":
    ex2()
    ex3()
