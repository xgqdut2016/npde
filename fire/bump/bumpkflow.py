from firedrake import *
import numpy as np

Re = 40
nu = 1/Re
mesh = Mesh('bump.msh')
x = SpatialCoordinate(mesh)


V = VectorFunctionSpace(mesh,'CG',2)*FunctionSpace(mesh,'CG',1)
solution = Function(V,name = 'pred')
testfunction = TestFunction(V)
u,p = split(solution)
v,q = split(testfunction)

lam = Constant(1/(2*nu) - sqrt(1/(4*nu*nu) + 4*pi*pi))
print(1/(2*nu) - sqrt(1/(4*nu*nu) + 4*pi*pi))
u_acc = as_vector([1 - exp(lam*x[0])*cos(2*pi*x[1]),lam*exp(lam*x[0])*sin(2*pi*x[1])/(2*pi)])
#print(mesh.coordinates.dat.data)

a = nu*inner(grad(u),grad(v))*dx + inner(dot(grad(u),u),v)*dx - p*div(v)*dx + div(u)*q*dx

DirBC = DirichletBC(V.sub(0),u_acc,[17,18,19,20])
parameter = {"snes_monitor": None,
                             "snes_rtol":1e-4,"snes_atol":1e-50,"snes_stol":1e-8,"snes_max_it":100,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"}
#parameter = {'ksp_type':'cg'}
solve(a == 0, solution, bcs=DirBC, solver_parameters=parameter)



x = mesh.coordinates.dat.data
u,p = solution.split()
u.rename("kflowu")
p.rename("kflowp")
File("kflow.pvd").write(u,p)
import matplotlib.pyplot as plt
fig, axes = plt.subplots()
levels = np.linspace(0, 1, 51)
contours = tricontourf(p, levels=levels, axes=axes, cmap="inferno")
axes.set_aspect("equal")
fig.colorbar(contours)
fig.show()
plt.savefig('kflow.png')
err = inner(u - u_acc,u - u_acc)*dx
print(err)
L1_err = assemble(abs(u[0] - u_acc[0])*dx)
print('the obj:%.3e'%(L1_err))
-- INSERT --                 

