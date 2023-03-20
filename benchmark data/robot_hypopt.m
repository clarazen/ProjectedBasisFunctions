clear; close all; clc; format compact
load X_sarcos.csv
load y_sarcos.csv
load Xtest_sarcos.csv
load ytest_sarcos.csv

Xall            = [Xtest';X'];
yall            = [ytest(1,:)';y(1,:)'];
D               = size(X,2);
Xall            = Xall + (max(Xall)-min(Xall))/2 - max(Xall);
Xall            = Xall ./ max(Xall);
yall            = yall - mean(yall);
%% training
Xtest           = Xall(1:size(Xtest,1),:);
ytest           = yall(1:size(Xtest,1));
X               = Xall(size(Xtest,1)+1:size(Xtest,1)+10000,:);
y               = yall(size(Xtest,1)+1:size(Xtest,1)+10000);

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, X, y);
%
l1  = 1
s_f = exp(hyp2.cov(2))
s_n = exp(hyp2.lik)

%% predictions
hyp2.cov(1) = 0;

[mu,s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, y, Xtest);

%%
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([Xtest; flipdim(Xtest,1)], f, [7 7 7]/8)
hold on; plot(Xtest, mu); plot(X, y, '+')

