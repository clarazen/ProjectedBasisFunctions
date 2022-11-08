using LinearAlgebra
using Plots
using StatsBase
using Pkg
# type ] add "https://github.com/clarazen/BigMat.git" 
# type ] add "https://github.com/clarazen/TN4GP.git"

Pkg.activate("C:/Users/cmmenzen/.julia/dev/BigMat")
using BigMat
Pkg.activate("C:/Users/cmmenzen/.julia/dev/TN4GP/")
using TN4GP


Nd       = 20;   # number of data points
D        = 2;      # dimensions
M        = 40;     # number of basis functions per dimension
hyp      = [.01*ones(D), 1.0, .5];
hyp_opt  = hyp
X,coord  = gengriddata(Nd,D,-1*ones(D),1*ones(D),true);
K        = covSE(X,X,hyp)
y        = cholesky(K+1e-10*Matrix(I,Nd^D,Nd^D)).L*randn(Nd^D)

# reduced-rank approach
boundsMin = minimum(X,dims=1);
boundsMax = maximum(X,dims=1);
L         = 2*((boundsMax.-boundsMin)./2)[1,:];
Mr        = 20
Φ,invΛ,S  = colectofbasisfunc(Mr*ones(D),X,hyp_opt[1],hyp_opt[2],L);
p         = sortperm(diag(mpo2mat(invΛ)));
Φmat      = khr2mat(Φ)*mpo2mat(S);

pl = Vector{Plots.Plot{Plots.GRBackend}}(undef,36)
for i = 1:36
    pl[i] = scatter(X[:,1],X[:,2],marker_z=Φmat[:,p[i]],markerstrokewidth = 0,markerwidth=.01,axis=([],false),legend=:none)
end
plot(pl...,layout=(6,6))
savefig("bf1")

# tt approach
Φ_        = colectofbasisfunc(M*ones(D),X,hyp_opt[1],hyp_opt[2],L,true);
Φ_mat     = khr2mat(Φ_);
norm(Φ_mat*Φ_mat'-K)/norm(K)

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
savefig("bf2")

