function [tap_f,tap_t] = recover_tap()

%filename = sprintf('data\\20180309_echo_taps\\echo.mat');
%filename = sprintf('VNA_RS\\data\\20180309_echo_taps\\echo.mat');
filename = sprintf('VNA_RS\\data\\20180313_echo_taps\\echo.mat');
temp = load(filename);
echo_f = temp.df;


%filename = sprintf('data\\20180309_echo_taps\\echo_m_tap.mat');
%filename = sprintf('VNA_RS\\data\\20180309_echo_taps\\echo_m_tap.mat');
filename = sprintf('VNA_RS\\data\\20180313_echo_taps\\echo_m_tap.mat');
temp = load(filename);
echo_m_tap_f = temp.df;

tap_f = -(echo_f - echo_m_tap_f);
tap_t = ifft(tap_f);
