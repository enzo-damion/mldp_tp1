import numpy as np

def genData():
    x = np.zeros([1000000, 2])
    y = np.zeros(1000000)
    cpt = 0
    for u_iter in np.arange(-5,5,0.01):
        for v_iter in  np.arange(-5,5,0.01):
            x[cpt,0] = u_iter
            x[cpt,1] = v_iter
            y[cpt] = np.sin(u_iter-v_iter)
            cpt+=1
    return x,y

def main():
    x,y = genData()

if __name__ == "__main__":
    main()