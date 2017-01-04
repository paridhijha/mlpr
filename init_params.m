function params = init_params(dim)
% initializes the row vector to be passed to slice_sample()
params=ones(1,dim+2);
params(1)=rand(); % 1st param is epsilon (e) - range btw 0 and 1
params(2)=rand(); % 2nd param is log(lambda) - range btw 0 and 1
b=9.376307381550576e+04; %  range of weight vector - random value
a=-7.460274498610420e+05;
for i = 3 : dim+2 % 3rd element till last element is weight vector
    params(i)=(b-a).*rand() + a;
end

