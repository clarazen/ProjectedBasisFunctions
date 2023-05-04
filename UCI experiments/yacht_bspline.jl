using LinearAlgebra
using Revise
using Optim
using Plots
using DelimitedFiles
using Metrics

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

X                   = Xtotn[1:247,:];
y                   = ytotn[1:247];
Xstar               = Xtotn[248:end,:];
ystar               = ytotn[248:end];
D                   = 6;
N                   = 308;

dd                  = 4;

ρ                   = 3
knotint             = 1
M                   = Int.((ρ + knotint)*ones(D))

maxiter             = 2;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xstar,ρ,knotint);

rnks                = Int.([1,4,16*ones(D-3)...,4,1]);
tt0                 = initTT(rnks,4,dd,D)
@time tt,covdd,res,ΦWd              = ALS_modelweights(y,Φ,maxiter,0.0,0.1,tt0);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)

# compute penalty matrix
P       = diff(I(ρ + knotint),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,dd,D,P,PP)
# Weighted sum of the difference penalty matrices
λ = 1e-6
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end
tt[dd]      = reshape(pinv(ΦWd'*ΦWd + WWW)*(ΦWd'*y),size(tt0[dd]))
norm(tt[dd])
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
r2_score(mstar,ystar)


# another ALS
@time tt,covdd,res,ΦWd              = ALS_modelweights(y,Φ,maxiter,1e-10,0.1,tt);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
(RMSE(mstar,ystar))^2
norm(mstar-ystar)/norm(ystar)

r2_score(mstar,ystar)