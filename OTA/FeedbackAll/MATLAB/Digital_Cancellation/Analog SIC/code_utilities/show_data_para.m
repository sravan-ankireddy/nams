% show data parallel 
% data = {data1,data2};
% str = {str1,str2};

function show_data_para(data,str)

if nargin < 2
  str = '';  
end

figure; 
for idx = 1:length(data)    
   x = 1:length(data{idx});
   plot(x,data{idx},'.-'); hold on;   
end

legend(str);
