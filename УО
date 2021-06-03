from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from sympy import Symbol
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


go = 'y'

l = 1
N = 0

znach_x = []
znach_y = []
znach_z = []
R = 10000

func_usage = 0
print("choose derrive method: ")
scheme_selection = 'm' #input()

def func_rosenbroks(x1, x2):
    global R
    global func_usage
    func_usage += 1
    #return ((1 - x1) ** 2) + (100 * (x2 - (x1 ** 2)) ** 2)
    return ((1 - x1) ** 2) + (100 * (x2 - (x1 ** 2)) ** 2) + R * (1 - (x1 ** 2) - (x2 ** 2))**2 + R * ((x1 ** 2) + (x2**2) - 0.5)**2
    #return ((1 - x1) ** 2) + (100 * (x2 - (x1 ** 2)) ** 2) + R * (1 - (x1 ** 2) - (x2 ** 2))**2

h = 0.00001



def dev_midle_func(x1, x2, s):
    if s == 'x':
        fs_x = (func_rosenbroks(x1 + h, x2) - func_rosenbroks(x1 - h, x2)) / (2 * h)
    elif s == 'y':
        fs_x = (func_rosenbroks(x1, x2 + h) - func_rosenbroks(x1, x2 - h)) / (2 * h)
    else:
        print('we have trouble')
    return fs_x


# x1 = Symbol('x1')
# x2 = Symbol('x2')
# z = (((1 - x1) ** 2) + (100 * (x2 - (x1 ** 2)) ** 2))
# x_d = z.diff(x1)
# y_d = z.diff(x2)

# print('x1_d = ', x_d)
# print('x2_d = ', y_d)


def d2z(x1, x2, s):
    if s == 'x':
        fs_x = (func_rosenbroks(x1 + h, x2) - (2 * func_rosenbroks(x1, x2)) + func_rosenbroks(x1 - h, x2)) / (h * h)
    elif s == 'y':
        fs_x = (func_rosenbroks(x1, x2 + h) - (2 * func_rosenbroks(x1, x2)) + func_rosenbroks(x1, x2 - h)) / (h * h)
    else:
        print('we have trouble')
    return fs_x


znach_z.append(func_rosenbroks(-1.2, 0))

X = [-1.2, 0]

znach_x.append(X[0])
znach_y.append(X[1])

x1 = Symbol('x1')
x2 = Symbol('x2')

y = ((1 - x1) ** 2) + (100 * (x2 - (x1 ** 2)) ** 2) + R * (1 - (x1 ** 2) - (x2 ** 2))**2 + R * ((x1 ** 2) + (x2**2) - 0.5)**2
# deltha = 2

x1d = y.diff(x1)
x2d = y.diff(x2)

k1x1_1 = 2
k1x1_3 = 400
k1x1x2 = -400
k1c = -2

k2x2 = 200
k2x1_2 = -200

print(x1d)

print(x2d)

H = [[x1d.diff(x1), x1d.diff(x2)],
     [x2d.diff(x1), x2d.diff(x2)]]

print('H = ')
print(H[0])
print(H[1])

counter = 1

while True:
    H_c = [[H[0][0].evalf(subs={x1: X[0], x2: X[1]}), H[0][1].evalf(subs={x1: X[0], x2: X[1]})],
           [H[1][0].evalf(subs={x1: X[0], x2: X[1]}), H[1][1].evalf(subs={x1: X[0], x2: X[1]})]]

    print('\nH_c = ')
    print(H_c[0])
    print(H_c[1])

    detH = (H_c[0][0] * H_c[1][1]) - (H_c[0][1] * H_c[1][0])
    print('---------------\ndetH = ', detH)

    trH = [[H_c[1][1], H_c[1][0]],
           [H_c[0][1], H_c[0][0]]]
    print('---------------\ntransposed H = ')
    print(trH[0])
    print(trH[1])

    turnH = [[(trH[0][0]/detH), (trH[0][1]/detH)],
             [(trH[1][0]/detH), (trH[1][1]/detH)]]
    print('---------------\nturned H = ')
    print(turnH[0])
    print(turnH[1])

    dY_midle = [dev_midle_func(X[0], X[1], 'x'), dev_midle_func(X[0], X[1], 'y')]
    print('dY_midle', dY_midle)

    dY = [(-400 * X[0] * (-X[0]**2 + X[1]) + 2 * X[0] - 2), (-200 * X[0]**2 + 200 * X[1])]
    print('dY', dY)

    my_dY = []


    if scheme_selection == 'm':
        my_dY = dY_midle
    if scheme_selection == 'o':
        my_dY = dY


    norm_grad = math.sqrt((my_dY[0] ** 2) + (my_dY[1] ** 2))
    print('---------------\n---------------\nnorm_grad', norm_grad)

    deltha_X = [((turnH[0][0] * my_dY[0]) - (turnH[0][1]*my_dY[1])), -((turnH[1][0] * my_dY[0]) - (turnH[1][1] * my_dY[1]))]

    print('deltha x ', deltha_X)

    X = [X[0] - l * deltha_X[0], X[1] - l * deltha_X[1]]

    znach_z.append(func_rosenbroks(X[0], X[1]))
    znach_x.append(X[0])
    znach_y.append(X[1])

    abs_error = math.fabs(znach_z[counter] - znach_z[counter - 1])
    rel_error = math.fabs(1 + ((znach_z[counter] - znach_z[counter - 1]) / math.fabs(znach_z[counter - 1]))) * 100
    counter += 1
    print('---------------\nAbsolute error = ', abs_error)
    print('Relative error = ', rel_error, '%')
    print('---------------\nNew X = ', X)
    print('---------------\nAll z= ', znach_z)
    print('All y = ', znach_y)
    print('All x = ', znach_x)
    if norm_grad<0.00001:
        print("norm grad succeed")
        break
    # if abs_error<0.00001 and rel_error<0.00001:
    #     print("abs and rel succeed")
    #     break
    # counter += 1


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.2)
Y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(X, Y)
Z = ((1-X)**2) + (100 * ((Y-X**2)**2))


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-50, 10000)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(znach_x, znach_y, znach_z, 'r-')
plt.show()

print(f"counter: {counter}")
print(f"func usages: {func_usage}")
