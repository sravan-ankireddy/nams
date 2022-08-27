function show_all_taps(taps_t)

taps_cell = cell(1,size(taps_t,1));
str = cell(1,size(taps_t,1));
for idx = 1:size(taps_t,1)
    taps_cell{idx} = taps_t(idx,:);
    str{idx} = num2str(idx); 
end
show_data_para(taps_cell,str);
  