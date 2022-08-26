function [eiter] = Linear_Block_NLMS(x,y,iter,post_cursor, pre_cursor,L)

M1 = pre_cursor;
M2 = post_cursor;

% Initializations
x = [zeros(M1,1);x;zeros(M2,1)];
wn = zeros(M1 + M2 + 1, 1);
eiter  = zeros(iter,1);
n = 1;
% muu = [.1 .02 .02 .02 .02 .02];
gn = 0;
mu = ones(33,1);
rho = .1;
for k = 1:fix(iter/L)
    s = 0;
    for j = 1:L    % Block Oper+ations
        xn= x( (k-1)*L + j: (k-1)*L + j + M1 + M2,1);
        yn = y((k-1)*L + j ,1);
        e = yn - conj(wn')*xn;
        eiter(n) = e;
        n = n+1;
        gnn = gn;
        gn = e.*conj(xn);
 mu = mu + rho*real(gn.*gnn);
 mu(mu>1) = 1;
 mu(mu<.0001) = .01;
 
%         mu(abs(gn)>0) = muu(1,6);
%         mu(abs(gn)>.01) = muu(1,5);
%         mu(abs(gn)>.03) = muu(1,4);
%         mu(abs(gn)>.05) = muu(1,3);
%         mu(abs(gn)>.08) = muu(1,2);
%         mu(abs(gn)>.1) = muu(1,1);

        %a = /(.001+norm(xn)^2);
        s = s+mu.*(e.*conj(xn));
    end
    wn = wn + (s/L);
end
end
