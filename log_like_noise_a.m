function [NLp, dLp_dw,dLp_da] = log_like_noise_a(ww, xx, yy)
yy = (yy==1)*2 - 1;
% this fn returns the -ve log likelihood for a model with noise
% unconstrained parameeter a and returns derivative wrt w and a
a = ww(end,:);
ww = ww(1:end-1,:);
e = 1/(1 + exp(-a));
sigmas = 1./(1 + exp(-yy.*(xx*ww)));
[Lp, dLp_dw] = log_like_noise_w(ww,xx,yy,e);
NLp=-Lp;
dLp_dw=dLp_dw*a;
dLp_da= sum(((log((1-log(a))*sigmas+(log(a)/2))).^(-1))*(((1/a)*sigmas)+(0.5*a)));



