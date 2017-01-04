clear
load('text_data.mat')
format 'long'


%% Bias feature
manyOnes1 = ones([length(x_train),1]);
Xtr = [x_train,manyOnes1];

manyOnes2 = ones([length(x_test),1]);
Xte = [x_test,manyOnes2];

%% Maximising likelihood

rng(1,'twister');
ww = rand(101,1);%;mlp_te = mean(log(1./(1 + exp(-Xte * optimised_weights.*y_test))));
optimised_weights = minimize(ww, @lr_loglike, 10, Xtr, y_train);

% probability of test and training set being classified as +1
yte = Xte * optimised_weights;
probability_yte = 1./(1 + exp(-yte));
ytr = Xtr * optimised_weights;
probability_ytr = 1./(1 + exp(-ytr));

% Accuracy

% assuming p = 0.5 => y_hat = +1
positives_te = (probability_yte >= 0.5);
negatives_te = -(probability_yte < 0.5);
all_predicted_te = positives_te + negatives_te;
testAccuracy = sum(y_test == all_predicted_te)/length(y_test);

positives_tr = (probability_ytr >= 0.5);
negatives_tr = -(probability_ytr < 0.5);
all_predicted_tr = positives_tr + negatives_tr;
trainingAccuracy = sum(y_train == all_predicted_tr)/length(y_train);

% Standard error
% Sqrt( Sum( (mean - x)^2 ) / (N-1) ) / Sqrt(N)

% Standard error of accuracy
accuracySE_te = standardError(y_test == all_predicted_te);
accuracySE_tr = standardError(y_train == all_predicted_tr);

% mean log probability and
% mean log probability standard error

mlp_te = mean(log(1./(1 + exp(-Xte * optimised_weights.*y_test))));
mlp_tr = mean(log(1./(1 + exp(-Xtr * optimised_weights.*y_train))));
mlpSE_te = standardError(log(1./(1 + exp(-Xte * optimised_weights.*y_test))));
mlpSE_tr = standardError(log(1./(1 + exp(-Xtr * optimised_weights.*y_train))));

% priors

priorPositive_te = sum(y_test > 0)/length(y_test);
priorNegativ_te = 1 - priorPositive_te;
priorPositive_tr = sum(y_train > 0)/length(y_train);
priorNegativ_tr = 1 - priorPositive_tr;

% comparing outputs ??

testStaticAccuracy = sum(y_test == ones(length(y_test),1))/length(y_test);
testMLPavgAccuracy = sum(y_test == -ones(length(y_test),1))/length(y_test);

%% Limited training data

% train on the first 100 data points
ww100 = rand(101,1);
optimised100_weights = minimize(ww100, @lr_loglike, 10, Xtr(1:100,:), y_train(1:100));

% probability of test set being classified as +1
yte100 = Xte * optimised100_weights;
probability_yte = 1./(1 + exp(-yte100));
mlp100_te = mean(log(y_test.*yte100));

%% Label Noise Model

%% Noise log probability

[lp, dw] = lr_loglike_noise_w(ww,Xtr,y_train,0.5);
gradW = checkgrad(@lr_loglike_noise_w, ww, 1e-5, Xtr, y_train, 0.5);
[lp, de] = lr_loglike_noise_e(0.5,Xtr,y_train,ww);
gradE = checkgrad(@lr_loglike_noise_e, 0.5, 1e-5, Xtr, y_train, ww);

%% Fitting a constrained parameter

% checking if gradients calculated are reasonable
epsilon = 1./(1 + exp(-0.5));

[lp, dw_c] = lr_loglike_noise_w(ww,Xtr,y_train,epsilon);
gradW = checkgrad(@lr_loglike_noise_w, ww, 1e-5, Xtr, y_train, epsilon);
[lp, de] = lr_loglike_noise_constrained_e(0.5,Xtr,y_train,ww);
gradA = checkgrad(@lr_loglike_noise_constrained_e, 0.5, 1e-5, Xtr, y_train, ww);

% adding a as the last feature of w
% optimising w

manyOnes1 = ones([length(x_train),1]);
Xtr = [x_train,manyOnes1,manyOnes1];
manyOnes2 = ones([length(x_test),1]);
Xte = [x_test,manyOnes2,manyOnes2];
rng(2,'twister');
ww_noise = rand(102,1);
optimised_noise_weights = minimize(ww_noise, @lr_loglike_noise_constrained_w, 10, Xtr, y_train);

% reporting e as required

e = 1./(1 + exp(-optimised_noise_weights(end)));

% mlp for noisy weights on initial model
mlp_noise_te = mean(log(1./(1 + exp(-Xte * optimised_noise_weights.*y_test))));

% probability of test set being classified as +1
yte_noise = Xte * optimised_noise_weights;
probability_yte = 1./(1 + exp(-yte_noise));


% Accuracy

% assuming p = 0.5 => y_hat = +1
positives_noise_te = (probability_yte >= 0.5);
negatives_noise_te = -(probability_yte < 0.5);
all_noise_predicted_te = positives_noise_te + negatives_noise_te;
test_noise_Accuracy = sum(y_test == all_noise_predicted_te)/length(y_test);

%% Hierarchical model & MCMC


