% show matrix row by row

function show_matrix(data,str)

if nargin < 2
  str = '';  
end

figure; 
for idx = 1:size(data,1)    
   x = 1:length(data(idx,:));
   plot(x,data(idx,:),'.-'); hold on;   
end

legend(str);
