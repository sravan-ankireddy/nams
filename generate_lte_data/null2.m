%Computes the null space of a binary matrix A over GF(2)
function [NullSpace]=null2(A)
A=mod(A,2);
%number of constraints:
m=size(A,1);
%number of variables:
n=size(A,2);
%number of independent constraints:
r=gfrank(A,2);
%Take care of the trivial cases:
if (r==n)
    NullSpace=[];
elseif (r==0)
    NullSpace=eye(n,n);
end
%Add one constraint at a time.
%Maintain a matrix X whose columns obey all constraints examined so far.
%Initially there are no constraints:
X=eye(n,n);
for i=1:m
    y=mod(A(i,:)*X,2);
    % identify 'bad' columns of X which are not orthogonal to y
    % and 'good' columns of X which are orthogonal to y
    GOOD=[X(:,find(not(y)))];
    %convert bad columns to good columns by taking pairwise sums
    if (nnz(y)>1)
      BAD=[X(:,find(y))];
      BAD=mod(BAD+circshift(BAD,1,2),2);
      BAD(:,1)=[];
    else
        BAD=[];
    end
    X=[GOOD,BAD];
end%for i
NullSpace=X;