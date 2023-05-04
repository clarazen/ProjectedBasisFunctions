using LinearAlgebra
using Revise
using Optim
using Plots
using DelimitedFiles
using SparseArrays
using StatsBase

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

X                   = readdlm("hypoptALS/X_cascaded.csv",',');
Xtest               = readdlm("hypoptALS/Xtest_cascaded.csv",',');
y                   = readdlm("hypoptALS/y_cascaded.csv",',')[:,1];
meany               = mean(y)
y                   = y .- meany
ystar               = readdlm("hypoptALS/ytest_cascaded.csv",',')[:,1];
ystar               = ystar .- meany                   
N,D                 = size(X)

M                   = Int.(10*ones(D))

rnks                = Int.([1,3,5*ones(D-3)...,3,1]);
maxiter             = 5;
coord               = gengriddata(M[1],D,0*ones(D),1*ones(D),false);
Wf_sp               = interpMatrix(X,coord,3);
Wstar_sp            = interpMatrix(Xtest,coord,3);

Wf                  = Vector{Matrix}(undef,D)
Wstar               = Vector{Matrix}(undef,D)
for d = 1:D
    Wf[d]           = Matrix(Wf_sp[d])
    Wstar[d]        = Matrix(Wstar_sp[d])
end

dd                  = 4
tt0                 = initTT(rnks,M[1],dd,D)
@time tt,covdd,res,ΦWd = ALS_modelweights(y,Wf,maxiter,1e-10,0.1,tt0);
res[end]
mstar               = khrtimesttm(Wstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
plot(ystar[33:end])
plot!(mstar)
ylims!((-.5,.5))

# compute penalty matrix
P       = diff(I(M[1]),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,dd,D,P,PP)
# Weighted sum of the difference penalty matrices

λ           = 1e-24
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end

tt[dd]      = reshape((ΦWd'*ΦWd + WWW)\(ΦWd'*y),size(tt[dd]))
norm(y - ΦWd*tt[dd][:])/norm(y)

mstar               = khrtimesttm(Wstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
plot(ystar[33:end])
plot!(mstar)
ylims!((-.5,.5))