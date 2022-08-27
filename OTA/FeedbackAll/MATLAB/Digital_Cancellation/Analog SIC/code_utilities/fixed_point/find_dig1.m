function y=find_dig1(x,base1,len_y)

y=zeros(len_y,1);
x_now=x;
q1=x;
q2=x;
for n=1:len_y        
    if q1 < base1
        y(len_y-n+1)=q1;
        break;
    else      
        x_now=q1;
        q1=floor(x_now/base1);
        q2=mod(x_now,base1);
        y(len_y-n+1)=q2;
    end            
end