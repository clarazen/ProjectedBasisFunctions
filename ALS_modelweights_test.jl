using LinearAlgebra
using Revise
using Pkg
Pkg.add("https://github.com/clarazen/BigMat.git")
using BigMat
Pkg.activate("https://github.com/clarazen/TN4GP.git")
using TN4GP
using Plots
using Optim
using DelimitedFiles
using StatsBase
using Distributions


N           = 5000;     # number of data points
D           = 3;       # dimensions
M           = 40;       # number of basis functions per dimension
hyp         = [0.01*ones(D), 1., 0.001];
Xall,~,~,~  = gensynthdata(N,D,hyp);

boundsMin   = minimum(Xall,dims=1);
boundsMax   = maximum(Xall,dims=1);
L           = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:] #.+ ((boundsMax.-boundsMin) ./ 4)[1,:];

Φ_          = colectofbasisfunc(M*ones(D),Xall,hyp[1],hyp[2],L,true);
Φ_mat       = khr2mat(Φ_);

K̃           = Φ_mat*Φ_mat';
K           = covSE(Xall,Xall,hyp);
            norm(K-K̃)/norm(K)

# test initialTT and getsupercore! for dir=1
R1          = 10
R2          = 11
R           = 10
tt0         = initialTT(D,M,[1,R1,R2,1]);
left,right  = initsupercores(Φ_,tt0);
test        = KhatriRao(left[2],Φ_[3],1);
test        = reshape(test,N,M,R2);
test        = permutedims(test,[1,3,2]);
test        = reshape(test,N,M*R2);
ref         = Φ_mat*mpo2mat(getWd(tt0,D));
norm(test-ref)/norm(ref)

# test getprojectedKhR
test = getprojectedKhR(3,left[2],randn(4,4,4),Φ_[3])

################
shiftMPTnorm(tt0,3,-1);
shiftMPTnorm(tt0,2,-1);
left,right  = initsupercores(Φ_,tt0,true);
test        = KhatriRao(right[2],Φ_[1],1);
ref         = Φ_mat*mpo2mat(getWd(tt0,1));
norm(test-ref)/norm(ref)

shiftMPTnorm(tt0,1,1);
left1        = Φ_[1]*reshape(tt0[1],M,R1)
right1       = Φ_[D]*reshape(tt0[D],R2,M)'
test        = KhatriRao(KhatriRao(right1,Φ_[2],1),left1,1)
ref         = Φ_mat*mpo2mat(getWd(tt0,2));
norm(test-ref)/norm(ref)


#
w1          = Matrix(qr(randn(M,R)).Q);
w2          = randn(R*M*R);
w3          = Matrix(qr(randn(M,R)).Q);
Wd          = kron(kron(w3,Matrix(I,M,M)),w1)

fall        = Φ_mat*Wd*w2
SNR         = 10;
noise_norm  = norm(fall)/(10^(SNR/20))
σ_n         = sqrt(noise_norm^2/(length(fall)-1))
noise       = randn(size(fall))
noise       = noise/norm(noise)*noise_norm;
y           = fall + noise

@time tt,res = ALS_modelweights(y,Φ_,[1,R1,R2,1],10,σ_n^2)

norm(Φ_mat*mps2vec(tt) - y)/norm(y)