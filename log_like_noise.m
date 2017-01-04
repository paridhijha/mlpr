function [Lp, dLp_dw,dLp_de] = log_like_noise(ww, xx, yy, e)
yy = (yy==1)*2 - 1;
% this fn returns the log likelihood for a model with noise e
sigmas = 1./(1 + exp(-yy.*(xx*ww)));
Lp = sum(log(((1-e)*sigmas)+(e/2)));
if nargout > 1
    dLp_dw = (1-e)*(((1./((1-e)*sigmas+e/2).*sigmas.*(1-sigmas)).*yy)' * xx)';
    dLp_de=  0;
    %sum((-1.*sigmas+1/2)*(1./((1-e)*sigmas+e/2)));
end
