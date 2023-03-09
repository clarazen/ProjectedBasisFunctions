clear; close all; clc; format compact
load X.csv
load y.csv
load Xtest.csv
load ytest.csv

Xall            = [Xtest';X'];
yall            = [ytest(1,:)';y(1,:)'];
D               = size(X,2);
Xall            = Xall + (max(Xall)-min(Xall))/2 - max(Xall);
Xall            = Xall ./ max(Xall);

%% training
Xtest           = Xall(1:3636,:);
ytest           = yall(1:3636);
X               = Xall(3637:13637,:);
y               = yall(3637:13637);

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, X, y);

l1  = exp(hyp2.cov(1))
s_f = exp(hyp2.cov(2))
s_n = exp(hyp2.lik)

%% predictions
[mu,s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, y, Xtest');

%%
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([Xtest'; flipdim(Xtest',1)], f, [7 7 7]/8)
hold on; plot(Xtest', mu); plot(X, y, '+')

