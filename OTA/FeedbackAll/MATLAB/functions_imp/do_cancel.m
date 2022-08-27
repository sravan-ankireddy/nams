function [e_iter, Cancel] = do_cancel(X,Y,type)
iter = 3e5;
x = X;
y = acorr(X,Y);
switch(type)
    case 'Linear'
        offset      = 18;
        post_cursor = 16;
        pre_cursor  = 16;
        Order       = 1;
        a           = .001;
        mu          = .5;
        [e_iter,~] = General_NLMS(x,y,offset,iter,Order,post_cursor,pre_cursor,a,mu);
        
    case 'Non-Linear'
        offset      = 18;
        post_cursor = 16;
        pre_cursor  = 16;
        Order       = 3;
        a           = .001;
        mu          = .5;
        [e_iter,~] = General_NLMS(X,y,offset,iter,Order,post_cursor,pre_cursor,a,mu);
end
Data_RX = 10*log10(movvar(y(1:iter),1000));
Result  = 10*log10(movvar(e_iter,1000));
Cancel  = Data_RX - Result;
end


