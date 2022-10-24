from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from slepc4py import SLEPc


msh = UnitSquareMesh(5,5,diagonal="crossed")
V = VectorFunctionSpace(msh, "CG", 4)
Q = FunctionSpace(msh, "DG", 3)
X = V*Q
u,p = TrialFunctions(X)
v,q = TestFunctions(X)

a = (inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx
m = inner(u,v)*dx

print("Problem Set Up !")

sol = Function(X)

print("Assembling ...")
bc = DirichletBC (X.sub(0), as_vector([0.0,0.0]) , [1,2,3,4]) # Boundary condition
A = assemble (a,bcs=bc)
M = assemble (m)
Asc, Msc = A.M.handle, M.M.handle
print("Assembled !")
print("Solving ...");

E = SLEPc.EPS().create()
E.setType(SLEPc.EPS.Type.ARNOLDI)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP);
E.setDimensions(5,SLEPc.DECIDE);
E.setOperators(Asc,Msc)
ST = E.getST();
ST.setType(SLEPc.ST.Type.SINVERT)
ST.setShift(12.0)
KSP = ST.getKSP()
KSP.setType("minres")
PC = KSP.getPC();
PC.setType("lu");
PC.setFactorSolverType("mumps");
E.setST(ST);
E.solve();
nconv = E.getConverged()
print("Number of converged eigenvalues is: {}".format(nconv))
for k in range(nconv): 
    vr, vi = Asc.getVecs()
    if k == 0:
        with sol.dat.vec_wo as vr:
            lam = E.getEigenpair(k, vr, vi)
    u,p = sol.split()
    File("SVish.pvd").write(u,p)
    print("Svaed !")
    lam = E.getEigenpair(k, vr, vi)
    print("[{}] Eigenvalue: {}".format(k,lam.real))

