function [Lp, dLp_dw] = lr_loglike_noise(ww, xx, yy, e)

yy = (yy==1)*2 - 1;

sigmas = 1./(1 + exp(-yy.*(xx*ww))); % Nx1
Lp = sum(log((1-e)*sigmas+(e/2)));

log_composite_derivative = 1./((1-e)*sigmas+e/2);



if nargout > 1
    dLp_dw = (1-e)*(((1./((1-e)*sigmas+e/2).*sigmas.*(1-sigmas)).*yy)' * xx)';
    if nargout > 2
        dLp_de = sum(sigmas + 0.5 + (1./((1-e)*sigmas+(e/2))))';
    end
end

% WARNING: The sigmas can numerically saturate to 1 for large weights. (Or zero,
% if the weights are a terrible explanation of the data). The gradient signal
% (1-sigmas) will then evaluate to zero. We could stave off problems for a
% little longer using Lp=-sum(log1p(exp(...))), and computing (1-sigmas) as
% 1./(1 + exp(yy.*(xx*ww))); However, it doesn't matter how careful we are,
% eventually the gradient *will* underflow for big enough weights. Netlab clips
% the input to the sigmoid functions to prevent total saturation, which could be
% considered. However, the way the computation is arranged here, nothing too bad
% happens (? -- at least we don't get NaNs!). The *real* problem is that models
% with sigmoids that can saturate so much are unrealistically confident. If we
% get numerical problems, our attention should really be on changing the models,
% so we can't get physically unrealistic probabilities in the first place.

