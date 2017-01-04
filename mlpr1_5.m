load welltrainedMLP.mat
load imgregdata.mat xte_nf yte_nf xtr_nf ytr_nf
% net struture is provided with data
ytr_pr = mlpfwd(net, xtr_nf);
yte_pr = mlpfwd(net, xte_nf);
rmse_tr = sqrt(mean((ytr_nf - ytr_pr).^2));  % RMSE on training set
rmse_te = sqrt(mean((yte_nf - yte_pr).^2));  % RMSE on test set
%part(b) - first 5000 data points 
arr_rmse_tr=[0 0 0 0 0];
arr_rmse_te=[0 0 0 0 0];
for i= [2015,2016,2017,2018,2019]
  rng(i,'twister')
  nhid = 10; % number of hidden units
  net = mlp(size(xtr_nf,2), nhid, 1, 'linear');% Set up the network
  options = zeros(1,18); % Set up vector of options for the optimiser.
  options(1) = 1; % This provides display of error values.
  options(9) = 1; % Check the gradient calculations.
  options(14) = 200; % Number of training cycles.
  % Train using scaled conjugate gradients.
  [net, options] = netopt(net, options, xtr_nf(1:5000,:), ytr_nf(1:5000,:), 'scg');
  ytr_pr = mlpfwd(net, xtr_nf);  
  yte_pr = mlpfwd(net, xte_nf);
  tr_rmse = sqrt(mean(((ytr_nf - ytr_pr).^2)));  % RMSE on training set
  te_rmse = sqrt(mean(((yte_nf - yte_pr).^2)));  % RMSE on test set
  arr_rmse_tr(i-2014)=tr_rmse;
  arr_rmse_te(i-2014)=te_rmse;
end
