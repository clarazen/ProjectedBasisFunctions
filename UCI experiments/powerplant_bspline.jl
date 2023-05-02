using LinearAlgebra
using Revise
using Optim
using Plots
using DelimitedFiles

includet("../functions/functionsBasic.jl")
using .functionsBasic
includet("../functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("../functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("../functions/functionsMetrics.jl")
using .functionsMetrics
includet("../functions/functionsTT.jl")
using .functionsTT
includet("../functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("../functions/functionsTTmatmul.jl")
using .functionsTTmatmul

X                   = Xtotn[1:7654,1:4];
y                   = ytotn[1:7654];
Xstar               = Xtotn[7655:end,1:4];
ystar               = ytotn[7655:end];
D                   = 4;
Md                  = 7;
N                   = 7654;

dd                  = 2;

ρ                   = 3
knotint             = 2
M                   = Int.((ρ + knotint)*ones(D))

maxiter             = 2;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xstar,ρ,knotint);

rnks                = Int.([1,4,16,4,1]);
@time tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,0.0,ρ,knotint,dd);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)

# compute penalty matrix
P       = diff(I(ρ + knotint),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,dd,D,P,PP)
# Weighted sum of the difference penalty matrices
λ = 1e-15
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end
tt[dd]      = reshape(pinv(ΦWd'*ΦWd + λ*WWW)*(ΦWd'*y),size(tt[dd]))
norm(tt[dd])
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)

tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,λ,ρ,knotint,dd);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)