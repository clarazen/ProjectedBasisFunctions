using DelimitedFiles
using StatsBase
using Revise
using LinearAlgebra
using Plots
using Optim

includet("functions/functionsBasic.jl")
using .functionsBasic
includet("functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("functions/functionsMetrics.jl")
using .functionsMetrics
includet("functions/functionsTT.jl")
using .functionsTT
includet("functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("functions/functionsTTmatmul.jl")
using .functionsTTmatmul
#includet("hypoptALS/logmarglik_pbf_exp_.jl")
#using .logmarklik

X               = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/X_kukainv.csv",','))
y               = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/y_kukainv.csv",','))[:,1]
Xtest           = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/Xtest_kukainv.csv",','))
ytest           = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/ytest_kukainv.csv",','))[:,1]

#ℓ²,σ_f²,σ_n²    = [.4706^2,2.8853^2,0.6200^2];
ℓ²,σ_f²,σ_n²    = [.1, 1, .1]
D               = size(X,2);

# compute basis functions per dimension
L               = ones(D) .+ 2*sqrt(ℓ²);
M               = 30*ones(D);
Φ_              = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_          = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

# test kernel matrix
Xsub            = X[1:10000,:];
Ksub            = covSE(Xsub,Xtest,[ℓ²,σ_f²,σ_n²])
K̃               = ones(10000,size(Ksub,2))
for d = 1:D
    K̃ = K̃ .* (Φ_[d][1:10000,:]*Φstar_[d]')
end
norm(K̃ - Ksub)/norm(Ksub)

rnks            = Int.([1,3*ones(D-1)...,1]);
maxiter         = 10;
dd              = 8

# first round with non optimized hyper parameters
tt,cov,res      = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,dd);
mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm             = getttm(tt,dd);  
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt            = Φstarttm*cov*Φstarttm';
s_tt            = sqrt.(diag(P_tt))

RMSE(mstar,ytest)
MSLL(mstar[:,1],s_tt,ytest,sqrt(σ_n²))
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[2s_tt 2s_tt])

# hyp opt with projected basis functions
Φ,sqrtΛ             = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
test                = logmarglik_pbf_exp(log.([ℓ²,σ_f²,σ_n²]),X,y,Φ,tt,M,L,dd)
hyp                 = [ℓ²,σ_f²,σ_n²]
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,M,L,dd)
optres              = optimize(obj,log.(hyp))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres))
logmarglik_pbf_exp(log.([ℓ²,σ_f²,σ_n²]),X,y,Φ_,tt,M,L,dd)

boundsMin           = minimum(X,dims=1);
boundsMax           = maximum(X,dims=1);
L                   = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*sqrt(ℓ²); 
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
Φstar_              = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

# second round with optimized hyperparameters
@time tt,cov,res = ALS_modelweights(y,Φ,rnks,maxiter,σ_n²,dd); 
mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm             = getttm(tt,dd);  
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt            = Φstarttm*cov*Φstarttm';
s_tt            = sqrt.(diag(P_tt))

RMSE(mstar,ytest)
MSLL(mstar[:,1],s_tt,ytest,sqrt(σ_n²))
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[2s_tt 2s_tt])