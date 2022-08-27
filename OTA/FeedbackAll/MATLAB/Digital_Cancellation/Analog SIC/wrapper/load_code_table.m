function  [code_table,code_to_coe_table]= load_code_table(params)

N_branches               = params.N_branches;
flag_normalize_coe_table = params.flag_normalize_coe_table;
flag_swap_tap3_tap8      = params.flag_swap_tap3_tap8;
flag_table_down_sample   = params.flag_table_down_sample;


code_to_coe_table        = cell(1,N_branches);

code_max    = 2^16;
N_codes     = 128;
code_table  = [1:floor(code_max/N_codes):code_max];


temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128','table_norm');
if flag_normalize_coe_table == 1
   temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
end

%------------- temp --------------
if flag_swap_tap3_tap8 == 1
   temp2 = temp.table_norm{3};
   temp.table_norm{3} = temp.table_norm{8};
   temp.table_norm{8} = temp2;   
end
%----------------------------------

if flag_table_down_sample == 1
    %===================================
    %           down sample
    %===================================
    R = 8; %8 is the maximum downsampling rate without compromising the performance
    table_down = cell(1,size(temp.table_norm,2));
    for idx = 1:size(temp.table_norm,2)
        temp_down = temp.table_norm{idx};
        table_down{idx} = temp_down(1:R:end);
    end
    code_to_coe_table{1} = table_down;    
    code_table = code_table(1:R:end);
else
    code_to_coe_table{1} = temp.table_norm;
end



temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128_external','table_norm');
% if flag_normalize_coe_table == 1
%    temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
% end
if flag_table_down_sample == 1
    %===================================
    %           down sample
    %===================================
    table_down = cell(1,size(temp.table_norm,2));
    for idx = 1:size(temp.table_norm,2)
        temp_down = temp.table_norm{idx};
        table_down{idx} = temp_down(1:R:end);
    end
    code_to_coe_table{2} = table_down;       
else
    code_to_coe_table{2} = temp.table_norm;
end