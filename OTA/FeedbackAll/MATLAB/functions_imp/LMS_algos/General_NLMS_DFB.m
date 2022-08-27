function [e_iter] = General_NLMS_DFB(x,y,offset,iter,order, pre_cursor, post_cursor,a,mu)

M1 = pre_cursor;
M2 = post_cursor;
K  = offset;
O  = order;

wn = zeros((M1 + M2 + 1)*O*(O+1)+32, 1);

e_iter  = zeros(iter,1);

for i = 33:iter
    
    for m = 1:O
        for n = 0:2*m-1
            xn(1:M1+M2+1,m*(m-1) + (n+1))=...
                ((x(K + (i-1) - M1:K + (i-1) + M2)).^n).*...
                ((conj(x(K + (i-1) -M1:K + (i-1) + M2))).^(2*m-1-n));
        end
    end
    xxn = xn(:);
    xxn = [e_iter(i-32:i-1,1);xxn];
    yn = y(K + (i-1));
    e_iter(i,1) = yn - wn'*xxn;
    step = mu/(a+ xxn'*xxn);
    wn = wn + step*conj(e_iter(i,1)).*xxn;
    
end

end

