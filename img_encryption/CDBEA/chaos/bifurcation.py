import numpy as np
from PWLCW import PWLCM
from matplotlib import pyplot as plt

def bifurcation_diagram(seed, n_skip, n_iter, step=1e-4, u_min=1e-16):
    '''
    绘制分岔图
    '''
    print("Starting with x0 seed {0}, skip plotting first {1} iterations, then plot next {2} iterations.".format(
        seed, n_skip, n_iter))
    # μ 列表, 分岔图的 x 轴
    U = []
    # x 列表, 分岔图的 y 轴
    X = []

    # Create the r values to loop. For each r value we will plot n_iter points
    u_range = np.linspace(u_min, 0.5-(1e-16), int(1/step))
    for u in u_range:
        U.extend([u]*(n_iter+1))
        X.extend(PWLCM(seed, u, n_iter+n_skip+1)[n_skip:])
    # X = [x / 2 for x in X]
    # Plot the data
    plt.figure(figsize=(3, 3), dpi=100)
    plt.title('bifurcation diagram of PWLCW')
    plt.plot(U, X, ls='', marker=',')
    plt.ylim(0, 1)
    plt.xlim(u_min, 0.5)
    plt.xlabel('μ')
    plt.ylabel('X')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bifurcation_diagram(0.2, 100, 10)
