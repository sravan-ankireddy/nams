% from quantize_r4 , remove y_digit

function [y y_digit]=quantize_simple(x,Num_bit_int,Num_bit_frac,flag_signed, flag_floor)
%=== input: 
%=== x: input signal, can be a vector and complex value
%=== Num_level: number level
%=== Max_A: range Max_A ~ Max_A-delta1
%=== range : -Max_A ~ Max_A-delta1
%=== Num_bit_int: number of integer bit
%=== Num_bit_frac: number of fractional bit
%=== flag_signed: input and output are signed
%=== flag_unsigned: input and output are unsinged

Num_bit=Num_bit_int+Num_bit_frac;

if isreal(x)
    [y ]=quantize_sub(x,Num_bit_int,Num_bit_frac,flag_signed, flag_floor);
else
    x_real=real(x);
    x_imag=imag(x);
    %=== real part ===================
    [y_tmp ] = quantize_sub(x_real,Num_bit_int,Num_bit_frac,flag_signed, flag_floor);
     y = y_tmp;
    %=================================
    
    %=== imag part ===================
    [y_tmp ] = quantize_sub(x_imag,Num_bit_int,Num_bit_frac,flag_signed, flag_floor);
     y = y + i*y_tmp;
    %=================================
end


function [y]=quantize_sub(x,Num_bit_int,Num_bit_frac,flag_signed, flag_floor)

Num_bit   = Num_bit_int+Num_bit_frac;
Num_level = 2^Num_bit;
N_tmp     = Num_level/2;

if flag_signed
    Max_A=2^(Num_bit_int-1);
    delta1=2*Max_A/Num_level;

    %=== clipping ================
    inx_tmp=find(x>Max_A);
    x(inx_tmp)=Max_A-delta1;
    inx_tmp=find(x<-Max_A);
    x(inx_tmp)=-Max_A;
    %=============================

    if flag_floor
        y_digit_tmp=floor((x+Max_A)/delta1);
        y=-Max_A+y_digit_tmp*delta1;
    else
        y_digit_tmp=round((x+Max_A)/delta1);
        y=-Max_A+y_digit_tmp*delta1;
    end
 
else
    %range 0 ~ Max_A
 
    Max_A=2^Num_bit_int;
    delta1=Max_A/Num_level;
    %=== clipping ================
    inx_tmp=find(x>Max_A);
    x(inx_tmp)=Max_A-delta1;
    inx_tmp=find(x<0);
    x(inx_tmp)=0;
    %=============================
    
    if flag_floor
        y_digit_tmp=floor(x/delta1);
        y=y_digit_tmp*delta1;
    else
        y_digit_tmp=round(x/delta1);
        y=y_digit_tmp*delta1;
    end
 
end




