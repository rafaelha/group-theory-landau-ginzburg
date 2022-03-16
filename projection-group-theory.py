import imp


import numpy as np

# 3d rep from Senechal
c4 = np.array([[ 0, 1, 0],
               [-1, 0, 0],
               [ 0, 0,-1]])
sx = np.array([[ 1, 0, 0],
               [ 0,-1, 0],
               [ 0, 0,-1]])
sy = np.array([[-1, 0, 0],
               [ 0, 1, 0],
               [ 0, 0,-1]])
sz = np.array([[-1, 0, 0],
               [ 0,-1, 0],
               [ 0, 0, 1]])

# E_g irep
# c4 = np.array([[0,1],[-1,0]])
# sx = np.array([[-1,0],[0,1]])
# sy = np.array([[1,0],[0,-1]])
# sz = np.array([[-1,0],[0,-1]])

e = sz@sz
c2 = c4@c4
c42 = c4.transpose()
i = sx@sy@sz
s4 = c4@sz
s42 = c42@sz
c2h = sx@sz
c2h2 = sy@sz
sd = c4@sx
sd2 = c42@sx
c2d = sd@sz
c2d2 = sd2@sz

cl = [[e], [c4,c42], [c2],[c2h,c2h2],[c2d,c2d2],[i],[s4,s42], [sz], \
    [sx,sy], [sd,sd2] ]


chars = np.array([np.trace(x[0]) for x in cl])
g = []
for c in cl:
    for gel in c:
        g.append(gel)
h = len(g)
g = np.array(g)
nj = np.array([len(x) for x in cl]) # size of each class
nc = len(cl) # number of classes (10 for D4h)

ctable = np.array([[1,1,1,1,1,1,1,1,1,1],
                   [1,1,1,-1,-1,1,1,1,-1,-1],
                   [1,-1,1,1,-1,1,-1,1,1,-1],
                   [1,-1,1,-1,1,1,-1,1,-1,1],
                   [2,0,-2,0,0,2,0,-2,0,0],
                   [1,1,1,1,1,-1,-1,-1,-1,-1],
                   [1,1,1,-1,-1,-1,-1,-1,1,1],
                   [1,-1,1,1,-1,-1,1,-1,-1,1],
                   [1,-1,1,-1,1,-1,1,-1,1,-1],
                   [2,0,-2,0,0,-2,0,2,0,0]])
ctable_all = np.zeros((nc, h))
i = 0
for ic in np.arange(len(cl)):
    for k in np.arange(len(cl[ic])):
        ctable_all[:,i] = ctable[:,ic]
        i += 1




irreps = ['A1g','A2g','B1g','B2g','Eg','A1u','A2u','B1u','B2u','Eu']

ax = np.newaxis
def decompose(chars):
    reps = 1/h * np.sum(nj[ax,:] * chars[ax,:] * ctable, axis=1)
    for i in np.arange(len(reps)):
        if reps[i]>0:
            if reps[i] - int(reps[i]) != 0:
                print('error, non integer factor of irrep contained')
            print(f'{int(reps[i])}*{irreps[i]}')
    return reps

# decompose(chars)
decompose(chars**4)


print(np.trace(g,axis1=1,axis2=2))
print(chars)

def sym(x):
    return 0.5*(x+x.transpose())
def asym(x):
    return 0.5*(x-x.transpose())

np.random.seed(1)
d = np.random.rand(9).reshape((3,3))
c4k = np.kron(c4,c4)
# print(np.transpose(c4)@d@c4 - (np.transpose(c4k)@np.reshape(d,((9,1)))).reshape((3,3)))
# print(c4@d@np.transpose(c4) - (c4k@np.reshape(d,((9,1)))).reshape((3,3)))

#%% projection of basis vectors

r = 1 # index of irrep
for r in range(nc):
    dr = ctable[r,0]
    gg = np.array([np.kron(g1,g1) for g1 in g])
    Pr = dr / h * np.sum( ctable_all[r,:][:,ax,ax]*gg, axis=0)
    def nm(x):
        m = np.max(x)
        if m != 0:
            return x/m
        else:
            return x

    np.random.seed(3)
    F = np.random.rand(9)
    print(irreps[r])
    # print('Pr=',Pr)
    res = nm(sym((Pr@F).reshape((3,3))))
    # res = nm((asym(Pr)@F).reshape((3,3)))
    print(res)


# for el in g: print(el.transpose()@res@c4)
#%% projection of basis vectors

kr = np.kron
for r in range(nc)[0:1]:
    dr = ctable[r,0]
    gg = np.array([kr(kr(kr(g1,g1),g1),g1) for g1 in g])
    Pr = dr / h * np.sum( ctable_all[r,:][:,ax,ax]*gg, axis=0)
    def nm(x):
        m = np.max(x)
        if m != 0:
            return x/m
        else:
            return x

    np.random.seed(3)
    F = np.random.rand(2**4)
    print(irreps[r])
    # print('Pr=',Pr)
    print(nm((Pr@F).reshape((2,2,2,2))))

    def print_term(z):
        string = ''
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        if z[i,j,k,l] == 1:
                            string += f'{i+1}{j+1}{k+1}{l+1}+'
        print(string[:-1])
        return string[:-1]

    z = nm((Pr@F).reshape((2,2,2,2)))
    print_term(z)
    for i in range(5):
        z[z==1] = 0
        z = nm(z)
        print_term(z)


#%%

from sympy import *

z = symbols("z")
z2 = symbols("z")

expr = Matrix(np.random.rand(4).reshape((2,2)))*(sin(z) +1)
expr.subs(z,-z)
