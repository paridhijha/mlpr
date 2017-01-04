load text_data.mat;
x_train = [x_train ones(size(x_train,1),1)]; %training and set expanded
x_test  = [x_test ones(size(x_test,1),1)]; % and bias feature added
%weights = ones(101,1);  % initialize using all 1's weight vector
weights=rand(101,1); % initialize randomly weight vector
dim=size(weights,1);
log_lambda=0.1; % initilize with random value btw 0 and 1
e=0.5; % initilize with random value btw 0 and 1
% handle of log_posterior passed through slice_sample
log_p=log_posterior(e,log_lambda,weights,dim,x_train,y_train);
% set the values for slice_sample
N=10; % no of iterations
burn=0; % default value
widths=1; % default value
step_out=false; % value set to true hangs the code.
rng(0,'twister');
init=init_params(dim); % row vector initialized
log_p_s = @(args) log_posterior(args{:}, dim, x_train, y_train);
% wrapper created so as to hide extra parameters
log_s_s = @(vector) log_p_s(split_params(vector));
% make sure that the initial row vector is inside the prob density area
assert(~isinf(log_s_s(init))); % before passing it to slice_sample
% MCMC slice sampling for heirarchial posterior
result = slice_sample(N, burn, log_s_s, init, widths, step_out);
% the result contains 3 params - e, lambda and w
params_cells=split_params(result);
e=params_cells(1);
log_lambda=params_cells(2);
weights=params_cells(3);
res_lambda=exp(log_lambda{1,1});
res_e=e{1,1};
res_weights=weights{1,1};
% scatter plot to confirm posterior belie about log lambda and e
plot(res_lambda,res_e,'r:+');
xlabel('lambda ?');
ylabel('epsilon e ');
title({'Plot a scatter plot of log ? against e'});
