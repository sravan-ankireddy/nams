function [RX] = acorr(X, Y)

[P1,~]=xcorr(X,Y);
[~,b1]=max(P1);
b1 = length(Y) - b1;
RX = Y(b1+1:b1+length(X));
end