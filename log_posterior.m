function log_post = log_posterior(e, log_lambda, ww, D, xx, yy)

if (e<=0 || e>1) % zero prior density
    log_post = -Inf;
    return;
end
if (log_lambda<=0 || log_lambda>1)
    log_post = -Inf;
    return;
end   
sigmas = 1./(1 + exp(-yy.*(xx*ww)));
lambda=exp(log_lambda);
% Dropped one constant term = -log(pi) 
log_prior = -lambda.*((ww')*ww) + (D/2)*log_lambda;
log_like = sum(log(((1-e)*sigmas)+(e/2)));
log_post = log_like + log_prior;
    
end







