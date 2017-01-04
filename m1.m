load text_data.mat;
x_train = [x_train ones(size(x_train,1),1)]; %training and set expanded
x_test  = [x_test ones(size(x_test,1),1)]; % and bias feature added
%weights = ones(101,1);  % initialize using all 1's weight vector
weights=rand(101,1); % initialize randomly weight vector

%create a negative-log-likelihood function
[NLp, dNLp_dw] = log_like_negative(weights, x_train, y_train);

% Minimize it given the training data x_train and y_train
minimized_weights = minimize(weights, @log_like_negative, 1, x_train, y_train);
bias_weight=weights(101); %bias weight

% using fitted weights, find the probability that y= +1 for each of the test 
% inputs x_test.
prob_te = 1./(1 + exp(-y_test.*(x_test*minimized_weights)));
% the probability is compared to y_test which has values(-1 or +1), and
% for prediction of label, prob >= 0.5, round is used to roundoff the prob.
prob_te_rd=round(prob_te);
acc_te=mean(prob_te_rd == y_test);
var_te=var(prob_te_rd == y_test);
std_err_te = sqrt(var_te/size(y_test,1));

% find mean log probability that predictions assign to test labels
mean_log_prob_te = mean(log(prob_te));

% performance of predictions on training set
prob_tr = 1./(1 + exp(-y_train.*(x_train*minimized_weights)));
prob_tr_rd=round(prob_tr);
acc_tr=mean(prob_tr_rd == y_train);
var_tr=var(prob_tr_rd == y_train);
std_err_tr = sqrt(var_tr/size(y_train,1));
mean_log_prob_tr = mean(log(prob_tr));

weights_ltd = rand(101,1);

% Fit the model with only the first N = 100 training cases
min_weights_ltd= minimize(weights_ltd, @log_like_negative, 100, x_train(1:100,:), y_train(1:100));
% probability of test set being classified as +1
prob_ltd = 1./(1 + exp(-y_train(1:100).*(x_train(1:100,:)*minimized_weights)));
avg_log_prob = (sum(log(prob_ltd)))/100;
