load text_data.mat;
x_train = [x_train ones(size(x_train,1),1)]; %training and set expanded
x_test  = [x_test ones(size(x_test,1),1)]; % and bias feature added
weights = ones(101,1);  % initialize using all 1's weight vector
%2.2(a) modifying the likelihood
[lp1, dw] = log_like_noise_w(weights,x_train,y_train,0.1);
grad_w = checkgrad(@lr_loglike_noise_w, weights, 1e-1, x_train, y_train, 0.1);
[lp2, de] = log_like_noise_e(0.1,x_train,y_train,weights);
grad_e = checkgrad(@lr_loglike_noise_e, 0.1, 1e-1, x_train, y_train, weights);
%2.2(b) fitting a constrained param : adding a 
weights = rand(102,1);
min_weights = minimize(weights, @log_like_noise_a, 100, x_train, y_train);
e = 1./(1 + exp(-min_weights(end)));
% mlp for noisy weights on initial model
mlp_noise_te = mean(log(1./(1 + exp(-x_test * min_weights.*y_test))));
% probability of test set being classified as +1
ytest = x_test * min_weights;
probability_yte = 1./(1 + exp(-ytest));


