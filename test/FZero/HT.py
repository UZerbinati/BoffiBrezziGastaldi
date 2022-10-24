from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from slepc4py import SLEPc

msh = UnitSquareMesh(20,20,quadrilateral=True)

V = VectorFunctionSpace(msh, "CG", 3)
Q = FunctionSpace(msh, "CG", 2)
X = V*Q
u,p = TrialFunctions(X)
v,q = TestFunctions(X)

a = (inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx+1e8*inner(u,v)*ds
m = inner(u,v)*dx

print("Problem Set Up !")

sol = Function(X)

bc = DirichletBC (X.sub(0), as_vector([0.0,0.0]) , [1,2,3,4]) # Boundary condition
print("Assembling ...")
A = assemble (a)
M = assemble (m)
Asc, Msc = A.M.handle, M.M.handle
print("Assembled !")
print("Solving ...");

E = SLEPc.EPS().create()
E.setType(SLEPc.EPS.Type.ARNOLDI)
E.setProblemType(SLEPc.EPS.ProblemType.GNHEP);
E.setDimensions(10,SLEPc.DECIDE);
E.setOperators(Asc,Msc)
ST = E.getST();
ST.setType(SLEPc.ST.Type.SINVERT)
ST.setShift(12.0)
PC = ST.getKSP().getPC();
PC.setType("lu");
PC.setFactorSolverType("mumps");
E.setST(ST);
E.solve();
nconv = E.getConverged()
print("Number of converged eigenvalues is: {}".format(nconv))
for k in range(nconv): 
    vr, vi = Asc.getVecs()
    lam = E.getEigenpair(k, vr, vi)
    print("[{}] Eigenvalue: {}".format(k,lam.real))
