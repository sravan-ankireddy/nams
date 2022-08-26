function [e_iter,wn] = do_cancel(X,Y)
iter = 5e5;
x = X;

offset      = 18;
post_cursor = 16;
pre_cursor  = 16;
Order       = 1;
a           = .001;
mu          = .5;
[e_iter,~,wn] = Linear_NLMS(x,y,offset,iter,Order,post_cursor,pre_cursor,a,mu);

end


