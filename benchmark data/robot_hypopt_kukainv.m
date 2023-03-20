clear; close all; clc; format compact

%% data pre-processing
load('inverse_identification_without_raw_data.mat')
Xall            = [u_train';u_test'];
yall            = [y_train(1,:)';y_test(1,:)'];
D               = size(Xall,2);

Xall            = Xall + (max(Xall)-min(Xall))/2 - max(Xall);
Xall            = Xall ./ max(Xall);
yall            = yall - mean(yall);

X               = Xall(1:size(u_train,2),:);
y               = yall(1:size(y_train,2),1);
Xtest           = Xall(size(u_train,2)+1:end,:);
ytest           = yall(size(y_train,2)+1:end);

writematrix(X,'X_kukainv.csv')
writematrix(y,'y_kukainv.csv')
writematrix(Xtest,'Xtest_kukainv.csv')
writematrix(ytest,'ytest_kukainv.csv')

%% training
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
% hyp2.cov(1) = 0;
% [mu,s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, X, y, Xtest);

%% plot
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
% fill([Xtest; flipdim(Xtest,1)], f, [7 7 7]/8)
% hold on; plot(Xtest, mu); plot(X, y, '+')

