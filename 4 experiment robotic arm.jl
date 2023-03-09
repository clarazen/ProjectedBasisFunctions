using DelimitedFiles
using StatsBase

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

X               = Matrix(readdlm("X.csv",',')');
y               = Matrix(readdlm("y.csv",',')')[:,1];
Xtest           = Matrix(readdlm("Xtest.csv",',')');
ytest           = Matrix(readdlm("ytest.csv",',')')[:,1];

ℓ²,σ_f²,σ_n²    = [0.1201^2,15.7048^2,0.01]#3.7821e-06^2];

# center data around 0 and normalize data to [-1,1] interval
D               = size(X,2);
Xall            = vcat(X,Xtest);
Xall            = Xall .+ (maximum(Xall,dims=1)-minimum(Xall,dims=1))/2 .- maximum(Xall,dims=1);
Xall            = Xall ./ maximum(Xall,dims=1);
X               = Xall[1:size(X,1),:];
Xtest           = Xall[size(X,1)+1:end,:];
yall            = vcat(y,ytest);
yall            = yall .- mean(yall);
y               = yall[1:size(y,1)];
ytest           = yall[size(y,1)+1:end];

L               = ones(D) .+ 2*sqrt(ℓ²);

M               = 20*ones(D);
Φ_              = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_          = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

R               = 20
rnks            = Int.([1, R*ones(D-1,1)..., 1]);
maxiter         = 5;
@time tt,res          = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²);

mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ytest)