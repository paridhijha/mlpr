load imgregdata.mat xte_nf yte_nf xtr_nf ytr_nf
% 1.2(a) - Subsampled by taking every other row from both datasets
% Both subsampled sets have roughly 8600 instances
xtr_ss=xtr_nf(1:2:end,:);
ytr_ss=ytr_nf(1:2:end,:);

figure 
plot(xtr_ss(:,end-34),xtr_ss(:,end),'r:+');

figure
scatter3(xtr_ss(:,end),xtr_ss(:,end-34),ytr_ss);
title({'2(a) Visualization - 3D plot of x(j, end), x(j, end ? 34) and y(j)'});
xlabel('x(j, end)');
ylabel('x(j, end - 34)');
zlabel('y(j)');

ytr=ytr_nf;
yte=yte_nf;

% 1.2(b) Denote the feature matrix as X, where each row is 3-dimensional, 
% the first 2 dimensions are x(j, end) and x(j, end ? 34), and the third is simply 1.
xtr=[(xtr_nf(:,end)) , (xtr_nf(:,end-34)) , ones((size(xtr_nf,1)),1)];
xte=[(xte_nf(:,end)) , (xte_nf(:,end-34)) , ones((size(xte_nf,1)),1)];
X=xtr;
y=ytr;
% max likelihood weight(w) = inv(X'*X)*X'*y
w=((X'*X)^(-1))*(X'*y);
% w = [0.46064 ; 0.52412 ; 0.00256]
ytr_pr=xtr*w;
yte_pr=xte*w;
% Squared errors
ytr_se=(ytr_nf-ytr_pr).^2;
yte_se=(yte_nf-yte_pr).^2;
% Root Mean of squared errors
rmse_tr=sqrt(mean(ytr_se));
rmse_te=sqrt(mean(yte_se));
% RMSE training = 0.0506 , RMSE test = 0.0503
% 1.2(c) Visualize the regression surface of this linear predictor in 3-D using Matlab function surf(). 
figure
[dim1,dim2] = meshgrid(0:0.01:1,0:0.01:1);
ysurf = [[dim1(:), dim2(:)], ones(numel(dim1),1)]*w;
surf(dim1, dim2, reshape(ysurf, size(dim1))); 
hold on;
scatter3(xte_nf(:,end),xte_nf(:,end-34), yte_nf);
xlabel('x(j, end)');
ylabel('x(j, end - 34)');
zlabel('y(j)');
title('1.2(c) Visualize - Regression surface of linear predictor, the 3-D plot of the test data points added');

