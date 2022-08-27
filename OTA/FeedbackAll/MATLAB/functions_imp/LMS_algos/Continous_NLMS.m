function [e_iter,wn,e_final] = Continous_NLMS(x,y,iter,order, pre_cursor, post_cursor,a,mu,wn)

M1 = pre_cursor;
M2 = post_cursor;
O  = order;


% Initialization
x  = [zeros(M1,1);x;zeros(M2,1)];
%wn = zeros((M1 + M2 + 1)*O*(O+1), 1);
pn = zeros((M1 + M2 + 1)*O*(O+1), iter);

e_iter  = zeros(iter,1);
e_final = zeros(iter,1);

for i = 1:iter
    for m = 1:O
        for n = 0:2*m-1
            xn(1:M1+M2+1,m*(m-1) + (n+1))=...
                ((x(i:i+M1+M2)).^n).*...
                ((conj(x(i:i+M1+M2))).^(2*m-1-n));
        end
    end
    xxn = xn(:);
    pn(:,i) = xxn;
    yn = y(i,1);
    e_iter(i,1) = yn - wn'*xxn;
    step = mu/(a+ xxn'*xxn);
    wn = wn + step*conj(e_iter(i,1)).*xxn;
    
end

for i = 1:iter
    e_final(i,1) = y(i,1) - wn'*pn(:,i);
end

end

