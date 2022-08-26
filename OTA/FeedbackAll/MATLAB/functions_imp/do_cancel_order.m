function [e_iter, Cancel, wn] = do_cancel_order(X,Y,order)
iter = 5e5;
y = acorr(X,Y);
offset      = 18;
post_cursor = 16;
pre_cursor  = 16;
Order       = order;
a           = .001;
mu          = .5;
[e_iter,wn] = General_NLMS(X,y,offset,iter,Order,post_cursor,pre_cursor,a,mu);
Data_RX = 10*log10(movvar(y(1:iter),1000));
Result  = 10*log10(movvar(e_iter,1000));
Cancel  = Data_RX - Result;
end


