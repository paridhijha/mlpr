clear
load('/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/imgregdata.mat')
addpath('/afs/inf.ed.ac.uk/user/s12/s1210107/mlpr/netlab3_3')
format long

%% Neural Network with all pixels
rng(2015,'twister')
% Set up the network
nhid = 10; % number of hidden units
net = mlp(size(xtr_nf,2), nhid, 1, 'linear');
% Set up vector of options for the optimiser.
options = zeros(1,18);
options(1) = 0; % This provides display of error values.
options(9) = 1; % Check the gradient calculations.
options(14) = 200; % Number of training cycles.
% Train using scaled conjugate gradients.
tic
[net, options] = netopt(net, options, xtr_nf(1:5000,:), ytr_nf(1:5000,:), 'scg');
toc
% RMSE on training set
ypred_tr = mlpfwd(net, xtr_nf);
rmse_NNsuball_tr = sqrt(mean(((ytr_nf - ypred_tr).^2)))
% RMSE on test set
ypred = mlpfwd(net, xte_nf);
rmse_NNsuball_te = sqrt(mean(((yte_nf - ypred).^2)))