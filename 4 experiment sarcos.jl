using DelimitedFiles
using StatsBase
using LinearAlgebra

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

X               = Matrix(readdlm("benchmark data/X_sarcos.csv",','))
y               = Matrix(readdlm("benchmark data/y_sarcos.csv",','))[:,1]
Xtest           = Matrix(readdlm("benchmark data/Xtest_sarcos.csv",','))
ytest           = Matrix(readdlm("benchmark data/ytest_sarcos.csv",','))[:,1]

ℓ²,σ_f²,σₙ²    = [1,24.3761^2,1.1083^2]; # sarcos

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

# compute basis functions per dimension
L               = ones(D) .+ 2*sqrt(ℓ²);
M               = 10*ones(D);
Φ_              = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_          = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

# test kernel matrix
Xsub            = X[1:10000,:];
Ksub            = covSE(Xsub,Xtest,[ℓ²,σ_f²,σₙ²])
K̃ = ones(10000,size(Ksub,2))
for d = 1:D
    K̃ = K̃ .* (Φ_[d][1:10000,:]*Φstar_[d]')
end
norm(K̃ - Ksub)/norm(Ksub)

rnks            = Int.([1,5*ones(20)...,1]);
maxiter         = 10;
dd              = 10;
@time tt,cov,res,tmp          = ALS_modelweights(y,Φ_,rnks,maxiter,σₙ²,dd);

mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm             = getttm(tt,dd);  
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt            = Φstarttm*cov*Φstarttm';
s_tt            = 2*sqrt.(diag(P_tt))
RMSE(mstar,ytest)
norm(mstar-ytest)/norm(ytest)

using Plots
plot(ytest[1:100])
plot!(mstar[1:100],ribbon=[s_tt s_tt])


# tests

A   = randn(8,8)
ttm,err = TTm_SVD(A,[2 2 2;2 2 2],0.0)
norm(ttm2mat(ttm)-A)

b   = randn(8)
ttm,err = TTm_SVD(reshape(b,8,1),[2 2 2; 1 1 1],0.0)
norm(ttm2mat(ttm))
norm(b)

tt,err = TTv_SVD(b,[2, 2, 2],0.0)
norm(ttv2vec(tt))



tt  = TT([randn(1,3,2),randn(2,3,2),randn(2,3,1)]);
khr = Vector{Matrix}(undef,3);
[khr[d] = randn(100,3) for d=1:3];
test = khrtimesttm(khr,tt2ttm(tt,[3 3 3;1 1 1]))[:,1];
ref  = khr2mat(khr)*ttv2vec(tt);
norm(test)
norm(ref)
