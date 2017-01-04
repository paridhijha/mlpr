load imgregdata.mat xte_nf yte_nf xtr_nf ytr_nf
% max likelihood weight(w) = inv(X'*X)*X'*y
weight = ((xtr_nf'*xtr_nf)^(-1))*(xtr_nf'*ytr_nf);
ytr_pr = xtr_nf*weight;
yte_pr = xte_nf*weight;
% Squared errors
ytr_se=(ytr_nf-ytr_pr).^2; 
yte_se=(yte_nf-yte_pr).^2;
% Root mean squared errors
rmse_tr=sqrt(mean(ytr_se)); % RMSE training = 0.0456
rmse_te=sqrt(mean(yte_se)); % RMSE test = 0.0371

