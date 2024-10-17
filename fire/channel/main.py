from firedrake import *
import numpy as np

Re = 50
gamma = 1/Re


mesh = Mesh('rflow.msh')
x = SpatialCoordinate(mesh)


V = VectorFunctionSpace(mesh,'CG',2)*FunctionSpace(mesh,'CG',1)
solution = Function(V,name = 'pred')
testfunction = TestFunction(V)
u,p = split(solution)
v,q = split(testfunction)


#print(mesh.coordinates.dat.data)

a = inner(grad(u),grad(v))*dx + Re*inner(dot(grad(u),u),v)*dx - p*div(v)*dx + div(u)*q*dx
uin = as_vector([(1 - x[1])*(1 + x[1]),0])
DirBC = [DirichletBC(V.sub(0),0,[17,18]),DirichletBC(V.sub(0),uin,14)]
parameter = {"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"}
solve(a == 0, solution, bcs=DirBC, solver_parameters=parameter)


out = File('rflow.pvd')

out.write(solution.split()[0])

x = mesh.coordinates.dat.data
u,p = solution.split()

err = assemble(abs(u[0] - uin[0])*dx)
print(err)
print('the obj:%.3e'%(err))
u = u.dat.data;p = p.dat.data
print(u.shape,p.shape)
print(x.shape,u.shape)
np.save('xx.npy',x)
np.save('u.npy',u)


