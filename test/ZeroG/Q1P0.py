from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from slepc4py import SLEPc

msh = SquareMesh(64,64,np.pi,quadrilateral=True)

S = VectorFunctionSpace(msh, "Q", 1)
V = FunctionSpace(msh, "DG", 0)
X = S*V
s,u = TrialFunctions(X); t,v = TestFunctions(X)

a = (inner(s,t)+inner(div(t),u)+inner(div(s),v))*dx
m = -inner(u,v)*dx

print("Problem Set Up !")

sol = Function(X)

bc = DirichletBC (X.sub(0), as_vector([0.0,0.0]) , [1,2,3,4]) # Boundary condition
print("Assembling ...")
A = assemble (a,bcs=bc)
M = assemble (m)
Asc, Msc = A.M.handle, M.M.handle
print("Assembled !")
print("Solving ...");

E = SLEPc.EPS().create()
E.setType(SLEPc.EPS.Type.ARNOLDI)
E.setProblemType(SLEPc.EPS.ProblemType.GNHEP);
E.setDimensions(25,SLEPc.DECIDE);
E.setOperators(Asc,Msc)
ST = E.getST();
ST.setType(SLEPc.ST.Type.SINVERT)
ST.setShift(1.0)
PC = ST.getKSP().getPC();
PC.setType("lu");
PC.setFactorSolverType("mumps");
E.setST(ST);
E.solve();
nconv = E.getConverged()
print("Number of converged eigenvalues is: {}".format(nconv))
for k in range(nconv): 
    vr, vi = Asc.getVecs()
    if k == 1:
        with sol.dat.vec_wo as vr:
            lam = E.getEigenpair(k, vr, vi)
            u,p = sol.split()
            u.rename("Vel")
            p.rename("Prs")
            File("Q1P0.pvd").write(u,p)
    lam = E.getEigenpair(k, vr, vi)
    print("[{}] Eigenvalue: {}".format(k,lam.real))
