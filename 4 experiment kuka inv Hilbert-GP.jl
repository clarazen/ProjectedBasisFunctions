using DelimitedFiles
using StatsBase
#using Revise
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

X               = Matrix(readdlm("benchmark data/X_kukainv.csv",','))
y               = Matrix(readdlm("benchmark data/y_kukainv.csv",','))[:,1]
Xtest           = Matrix(readdlm("benchmark data/Xtest_kukainv.csv",','))
ytest           = Matrix(readdlm("benchmark data/ytest_kukainv.csv",','))[:,1]

ℓ²,σ_f²,σ_n²    = [.4706^2,2.8853^2,0.6200^2];
D               = size(X,2);

# compute basis functions per dimension
L               = ones(D) .+ 2*sqrt(ℓ²);
M               = 30*ones(D);

budget          = size(X,1);
ΦR,ΛR           = colectofbasisfunc(budget,X,ℓ²,σ_f²,L)
ΦsR,ΛsR         = colectofbasisfunc(budget,Xtest,ℓ²,σ_f²,L)

Z               = σ_n²*diagm(1 ./ diag(Λ)) + ΦR'*ΦR;
Lchol           = cholesky(Z).L
mstar           = ΦsR'*(Lchol' \ (Lchol\ (ΦR'*y)))
vstar           = σ_n² * ΦsR'*(Lchol' \ (Lchol\ (ΦsR)))

RMSE(mstar,ytest)
MSLL(mstar[:,1],sqrt.(vstar),ytest,sqrt(σ_n²))
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[2sqrt.(vstar) 2sqrt.(vstar)])