from firedrake import *

sigma1 = 1
sigma2 = 2
#mesh = Mesh('immersed_domain.msh')
mesh = Mesh('test1.msh')
V = FunctionSpace(mesh,'CG',1)

u = TrialFunction(V)
v = TestFunction(V)
a = sigma2*dot(grad(v),grad(u))*dx(4) + sigma1*dot(grad(u),grad(v))*dx(3) + u*v*dx
L = Constant(5.)*v*dx + Constant(3.)*v('+')*dS(13)

Dirbc = DirichletBC(V,0,[11,12])

u = Function(V)
solve(a == L,u,bcs = Dirbc,solver_parameters = {'ksp_type':'cg'})
File('face.pvd').write(u)


