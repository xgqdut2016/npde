from firedrake import *

mesh = Mesh('bump.msh')
x = SpatialCoordinate(mesh)


V = FunctionSpace(mesh,'CG',1)
#print(mesh.coordinates.dat.data)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)

f.interpolate((pi*pi - 1)*sin(pi*x[0])*exp(x[1]))
g_o = Function(V).interpolate(-pi*exp(x[1]))
g_i = Function(V).interpolate(pi*exp(x[1]))
h = Function(V)
h.interpolate(exp(x[1])*sin(pi*x[0]))
a = dot(grad(u),grad(v))*dx

L = f*v*dx + g_i*v*ds(17) + g_o*v*ds(18)
DirBC = DirichletBC(V,h,[19,20])
u = Function(V)
solve(a == L, u, bcs=DirBC, solver_parameters={'ksp_type': 'cg'})
u_acc = Function(V).interpolate(exp(x[1])*sin(pi*x[0]))
L1_err = assemble(abs(u - u_acc)*dx)
print('firedrake solve poisson:L1_err:%3e'%(L1_err))
File('bump.pvd').write(u)


