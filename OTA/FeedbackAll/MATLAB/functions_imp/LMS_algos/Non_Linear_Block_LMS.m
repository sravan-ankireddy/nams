function [e_iter,e_final,wn] = Non_Linear_Block_LMS(x,y,step,Order, iter,post_cursor, pre_cursor,Block_Length)

L  = Block_Length;
M1 = pre_cursor;
M2 = post_cursor;

% Initialization
x  = [zeros(M1,1);x;zeros(M2,1)];
wn = zeros((M1 + M2 + 1)*2, 1);
pn = zeros((M1 + M2 + 1)*2, iter);
yn = zeros(iter,1);

e_iter  = zeros(iter,1);
e_final = zeros(iter,1);
wn_f = wn;
for i = 1:iter
    xn(1:M1+M2+1,1) = x(i:i + M1 + M2);
    xn(1:M1+M2+1,2) = x(i:i + M1 + M2).*x(i:i + M1 + M2).*conj(x(i:i + M1 + M2));
    xxn         = xn(:);
    pn(:,i)     = xxn;
    yn(i)       = y(i,1);
    e_iter(i,1)   = yn(i) - conj(wn_f')*xxn;
    wn = wn + (step/L)*e_iter(i,1).*conj(xxn);
    if (mod(i,L) == 0)
        wn_f = wn;
    end
end

for i = 1:iter
    e_final(i,1) = yn(i) - conj(wn')*pn(:,i);
end
end