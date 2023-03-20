using DelimitedFiles
using StatsBase
using Revise
using LinearAlgebra
using Plots

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

X               = Matrix(readdlm("benchmark data/X_kuka.csv",','))
y               = Matrix(readdlm("benchmark data/y_kuka.csv",','))[:,1]
Xtest           = Matrix(readdlm("benchmark data/Xtest_kuka.csv",','))
ytest           = Matrix(readdlm("benchmark data/ytest_kuka.csv",','))[:,1]

plot(ytest)

ℓ²,σ_f²,σ_n²    = [1^2,22^2,0.5935^2];

# compute basis functions per dimension
D               = size(X,2)
L               = ones(D) .+ 2*sqrt(ℓ²);
M               = 20*ones(D);
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

rnks            = Int.([1,10*ones(D-1)...,1]);
maxiter         = 10;
dd              = 5
tt,cov,res      = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,dd);
mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm             = getttm(tt,dd);  
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt            = Φstarttm*cov*Φstarttm';
s_tt            = 2*sqrt.(diag(P_tt))

RMSE(mstar,ytest)
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[s_tt s_tt])
