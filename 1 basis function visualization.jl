using LinearAlgebra
using Plots
using StatsBase
using Pkg
Pkg.activate("C:/Users/cmmenzen/.julia/dev/BigMat")
using BigMat
Pkg.activate("C:/Users/cmmenzen/.julia/dev/TN4GP/")
using TN4GP

# this experiment plots the dominating basis functions for the reduced-rank approach by Solin and Sarkaa,
# as well as the projected basis functions from our algorithm for a 2D case with input points on a grid

Nd       = 20;      # number of data points
D        = 2;      # dimensions
M        = 40;     # number of basis functions per dimension
hyp      = [.01*ones(D), 1.0, .5];
hyp_opt  = hyp
X,coord  = gengriddata(Nd,D,-1*ones(D),1*ones(D),true);

# reduced-rank approach
boundsMin = minimum(X,dims=1);
boundsMax = maximum(X,dims=1);
L         = 2*((boundsMax.-boundsMin)./2)[1,:];
Mr        = 40
Φ,invΛ,S  = colectofbasisfunc(Mr*ones(D),X,hyp_opt[1],hyp_opt[2],L);
p         = sortperm(diag(mpo2mat(invΛ)));
Φmat      = khr2mat(Φ)*mpo2mat(S);

pl = Vector{Plots.Plot{Plots.GRBackend}}(undef,36)
for i = 1:36
    pl[i] = scatter(X[:,1],X[:,2],marker_z=Φmat[:,p[i]],markerstrokewidth = 0,markerwidth=.01,axis=([],false),legend=:none)
end
plot(pl...,layout=(6,6))
#savefig("Plots/bf1")

# tt approach
Φ_        = colectofbasisfunc(M*ones(D),X,hyp_opt[1],hyp_opt[2],L,true);
Φ_mat     = khr2mat(Φ_)
K         = covSE(X,X,hyp)
norm(Φ_mat*Φ_mat'-K)/norm(K)

R           = 5;
w1          = Matrix(qr(randn(M,R)).Q)
w2          = randn(R*M)
Wd          = kron(Matrix(I,M,M),w1)

fall        = Φ_mat*Wd*w2
SNR         = 10;
noise_norm  = norm(fall)/(10^(SNR/20))
σ_n         = sqrt(noise_norm^2/(length(fall)-1))
noise       = randn(size(fall))
noise       = noise/norm(noise)*noise_norm;
y           = fall + noise

rnks    = Int.([1, 10*ones(D-1,1)..., 1]);
maxiter = 4;

# compute the mean with ALS
@time tt,res  =  ALS_modelweights(y,Φ_,rnks,maxiter,hyp[3]);
ttm2 = getU(tt,2);
tt   = shiftMPTnorm(tt,2,-1);
ttm1 = getU(tt,1); 
    
W2   = mpo2mat(ttm2);
U2   = krtimesttm(Φ_,transpose(ttm2));

pl = Vector{Plots.Plot{Plots.GRBackend}}(undef,36)
for i = 1:36
    pl[i] = scatter(X[:,1],X[:,2],marker_z=U2[i,:],markerstrokewidth = 0,axis=([],false),legend=:none)
end
plot(pl...,layout=(6,6))
#savefig("Plots/bf2")

