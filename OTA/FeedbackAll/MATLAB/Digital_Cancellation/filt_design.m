function b = filt_design(Fs,Fpass, Fstop)
%FILT_DESIGN Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.3 and DSP System Toolbox 9.5.
% Generated on: 13-May-2018 04:00:21

% Equiripple Lowpass filter designed using the FIRPM function.

% All frequency values are in MHz.
            
Dpass = 0.057501127785;  % Passband Ripple
Dstop = 0.0001;          % Stopband Attenuation
dens  = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);

% Calculate the coefficients using the FIRPM function.
b  = cfirpm(N, Fo, Ao, W, {dens});
%Hd = dfilt.dffir(b);

% [EOF]
