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

X                   = Xtotn[1:1280,:];
y                   = ytotn[1:1280];
Xstar               = Xtotn[1281:end,:];
ystar               = ytotn[1281:end];
D                   = 11;
N                   = 1599;

dd                  = 4;

# FIND OUT WHY YHE EIGENVALUES ARE SO LARGE! 
# Bayesian update with scaled basis functions

ν                   = 7/2
ℓ²,σ_f²,σ_y²        = [.1, 1., 0.01];
L                   = ones(D) .+ 2*sqrt(ℓ²);
Md                  = 30
M                   = Md*collect(ones(D))
Φ                   = Vector{Matrix}(undef,D);
Λ                   = Vector{Vector}(undef,D);
                    basisfunctionsMatern!(M,X,ℓ²,σ_f²,L,Φ,Λ,ν);
Φstar               = Vector{Matrix}(undef,D);
Λstar               = Vector{Vector}(undef,D);
                    basisfunctionsMatern!(M,Xstar,ℓ²,σ_f²,L,Φstar,Λstar,ν);

rnks                = Int.([1,10,12*ones(D-3)...,10,1]);
@time tt,covdd,res,ΦWd    = ALS_modelweights(y,Φ,initTT(rnks,Md,dd,D),3,1e-10,σ_y²,2)
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)

# use prior
ℓ²,σ_f²,σ_y²        = [.5, 1, 1e-10];
WdΛWd               = projectedprior(tt,Λ,dd)
tt[dd]              = reshape((ΦWd'*ΦWd + σ_y²*WdΛWd)\(ΦWd'*y),size(tt[dd]))
norm(tt[dd])
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)