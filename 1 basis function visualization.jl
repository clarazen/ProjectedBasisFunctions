using LinearAlgebra
using StatsBase
using Plots

includet("functions/functionsBasic.jl")
using .functionsBasic
includet("functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("functions/functionsTT.jl")
using .functionsTT
includet("functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("functions/functionsTT.jl")
using .functionsTT
includet("functions/functionsTTmatmul.jl")
using .functionsTTmatmul

# this experiment plots the dominating basis functions for the reduced-rank approach by Solin and Sarkaa,
# as well as the projected basis functions from our algorithm for a 2D case with input points on a grid

Nd       = 20;      # number of data points
D        = 2;      # dimensions
Md       = 40;     # number of basis functions per dimension
hyp      = [.01, 1.0, .01];

X        = hcat((range(-1,1,Nd)' .* ones(Nd))[:], (ones(Nd)' .* range(-1,1,Nd))[:])

# reduced-rank approach
boundsMin = minimum(X,dims=1);
boundsMax = maximum(X,dims=1);
L         = ((boundsMax.-boundsMin)./2 .+ 2*sqrt(hyp[3]))[1,:]
ΦR,ΛR     = colectofbasisfunc(M^D,X,hyp[1],hyp[2],L);

pl = Vector{Plots.Plot{Plots.GRBackend}}(undef,32)
for i = 1:32
    pl[i] = scatter(X[:,1],X[:,2],marker_z=ΦR*sqrt.(ΛR)[:,i],markerstrokewidth = 0,markerwidth=.01,axis=([],false),legend=:none)
end
plot(pl...,layout=(4,8))
savefig("plots/bf1")

# tt approach
Φ_        = colectofbasisfunc(Md*ones(D),X,hyp[1],hyp[2],L);
Φ_mat     = khr2mat(Φ_)
K         = covSE(X,X,hyp)
norm(Φ_mat*Φ_mat'-K)/norm(K)

R           = 5;
w1          = Matrix(qr(randn(M,R)).Q)
w2          = randn(R*M)
Wd          = kron(Matrix(I,M,M),w1)

fall        = Φ_mat*Wd*w2
SNR         = 40;
noise_norm  = norm(fall)/(10^(SNR/20));
σ_n         = sqrt(noise_norm^2/(length(fall)-1));
noise       = randn(size(fall));
noise       = noise/norm(noise)*noise_norm;
y           = fall + noise;

rnks        = Int.([1, 5*ones(D-1,1)..., 1]);
maxiter     = 4;

# compute the mean with ALS
tt,res      =  ALS_modelweights(y,Φ_,rnks,maxiter,hyp[3]);
ttm2        = getttm(tt,2);   
W2          = ttm2mat(ttm2);
U2          = khrtimesttm(Φ_,ttm2);

pl          = Vector{Plots.Plot{Plots.GRBackend}}(undef,32)
for i = 1:32
    pl[i] = scatter(X[:,1],X[:,2],marker_z=(U2)[:,i],markerstrokewidth = 0,axis=([],false),legend=:none)
end
plot(pl...,layout=(4,8))
savefig("plots/bf2")

