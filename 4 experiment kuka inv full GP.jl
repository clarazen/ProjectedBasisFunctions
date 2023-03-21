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

mstar,vstar     = fullGP(covSE(X,X,[ℓ²,σ_f²,σ_n²]),X,Xtest,y,[ℓ²,σ_f²,σ_n²])

RMSE(mstar,ytest)
MSLL(mstar[:,1],sqrt(vstar),ytest,sqrt(σ_n²))
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[2sqrt(vstar) 2sqrt(vstar)])