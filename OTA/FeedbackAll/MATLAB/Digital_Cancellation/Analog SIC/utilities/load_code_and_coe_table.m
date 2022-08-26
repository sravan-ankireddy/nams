function  [code_table,code_to_coe_table] = load_code_and_coe_table(N_branches)

folder_name = '..\\data\\20180704_cc3_pcb3';
folder_name = '..\\data\\20180705_cc3_pcb3_IA';
filename = sprintf('%s\\code_table_256.mat',folder_name);
temp = load(filename);
code_table = temp.code_table;

%% code to coe
code_to_coe_table        = cell(1,N_branches);
filename                 = cell(1,N_branches);
filename{1} = sprintf('%s\\vna_cm_code_to_coe_N256',folder_name);
%filename{2} = sprintf('%s\\vna_cm_code_to_coe_N256',folder_name);

for idx = 1:N_branches
    temp = load(filename{idx},'table_norm');
    code_to_coe_table{idx} = temp.table_norm;
end

%-------- from Ian's measurement --------
if 0
    folder_name = '..\\data\\20180621_cc3_pcb3';
    filename = sprintf('%s\\tapdata.csv',folder_name);
    temp = load(filename);
    code_table = temp(:,1);
    
    % normalize each taps
    N_taps = 7;
    table_norm = cell(1,1);
    for idx = 1:N_taps
        temp_curve = temp(:,idx+1)./temp(1,idx+1);
        table_norm{idx} = temp_curve;
        %----------- test ------------
        %table_norm{idx} = temp_curve.^2;
        %-------------------------------
    end
    
    % code to coe
    code_to_coe_table        = cell(1,N_branches);
    for idx = 1:N_branches
        code_to_coe_table{idx} = table_norm;
    end
end
%----------------------------------------
