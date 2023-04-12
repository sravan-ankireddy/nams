% clearvars;
% clc;
% Parameters


code = "BCH"; % BCH/LDPC/Conv/Turbo or Polar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_frames = 20;
no_of_ofdm_symbols_per_frame = 4500;
no_signal_symbols = 1;
total_ofdm_symbols_per_frame = no_of_ofdm_symbols_per_frame + no_signal_symbols;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_msg_symbols = no_of_ofdm_symbols_per_frame * no_of_subcarriers;
signal_field_symbols = no_signal_symbols * no_of_subcarriers;
mod_order = 2;
bit_per_symbol = log2(mod_order);
total_no_bits = total_msg_symbols * bit_per_symbol;

if (code == "BCH")
    load('data_files/par_gen_data/G_BCH_63_36.mat','G');
    load('data_files/par_gen_data/H_BCH_63_36.mat','H');
else
    load('data_files/par_gen_data/G_LDPC_384_320.mat','G');
    load('data_files/par_gen_data/H_LDPC_384_320.mat','H');
end

G = double(G);
H = double(H);
msg_len = size(G,2);
code_len = size(G,1);

% creating sparse logical version of H
Hs = sparse(logical(H));

% create comm objects
ldpcEncCfg = ldpcEncoderConfig(Hs);

% create comm objects    
ldpcDecCfg = ldpcDecoderConfig(Hs,'norm-min-sum');
max_iter = 5;

% Polar params
if (code == "Polar")
    msg_len = 16;
    code_len = 32;
end

rate = msg_len/code_len;

% Turbo code config
if (code == "Turbo")
    enc_type = 'turbo'; %'convolutional'
    dec_type = 'turbo'; %'convolutional' 'MAP'
elseif (code == "Conv")
    enc_type = 'convolutional';
    dec_type = 'convolutional';
end

if (code == "Turbo" || code == "Conv")
    msg_len = 200; % Convolutional Code Parameter
    term_bits = 4;
    rate = 1/3;
    code_len = (msg_len + term_bits) / rate;
end

no_of_blocks = floor(total_no_bits / code_len);%(block_len + term_bits));
encoded_no_bits = msg_len * no_of_blocks;
no_encoder_out_bits = code_len * no_of_blocks;
extra_bits = total_no_bits - no_encoder_out_bits;
no_preamble_symbols = 4;
preamble_len = no_preamble_symbols * (size_of_FFT + cp_length);
total_no_of_data_samples = total_ofdm_symbols_per_frame * (size_of_FFT + cp_length);
total_no_of_samples = total_no_of_data_samples + preamble_len;
no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (code == "Polar")
    rs = [256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ...
    ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ...
    ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ...
    ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ...
    ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ...
    ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ...
    ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ...
    ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ...
     ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ...
      ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ...
      ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ...
      ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ...
      ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ];
end
