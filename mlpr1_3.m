load imgregdata.mat xte_nf yte_nf xtr_nf ytr_nf
xtr_nf=[xtr_nf(:,end), xtr_nf(:,end-34)];
xte_nf=[xte_nf(:,end), xte_nf(:,end-34)];
options = foptions; % nbf is the number of basis functions
options(1) = 1; % Display EM training 
options(14) = 5; % number of iterations of EM 
count=1;
rmse = [0,0,0,0,0];
n_rbf = [5,10,15,20,25];
while (count <= 5)% nbf using the candidates {5, 10, 15, 20, 25, 30}.
rbf_fn=@(xtr,ytr,xte)(rbffwd(rbftrain((rbf(2, n_rbf(count), 1, 'gaussian')),options,xtr,ytr),xte));
rmse(count) = sqrt(crossval('mse',xtr_nf,ytr_nf,'Predfun',rbf_fn));
count=count+1;
end
figure
plot(n_rbf,rmse,'b+:');
xlabel('number of RBFs used');
ylabel('cross-validation RMSE for the training set');
title('1.3(a)  Plot of the cross-validation RMSE against the number of RBFs used');
net_tr = rbf(2, 10, 1, 'gaussian');% 1.3(b)  best model nbf = 10 
net_tr = rbftrain(net_tr,options,xtr_nf,ytr_nf);
ytr_pr = rbffwd(net_tr,xtr_nf);
ytr_se=(ytr_nf-ytr_pr).^2;
rmse_tr=sqrt(mean(ytr_se)); % rmse tr = 0.0498 
net_te = rbf(2, 10, 1, 'gaussian');
net_te = rbftrain(net_te,options,xte_nf,yte_nf);
yte_pr = rbffwd(net_te,xte_nf);
yte_se=(yte_nf-yte_pr).^2;
rmse_te=sqrt(mean(yte_se)); % rmse te = 0.0504
