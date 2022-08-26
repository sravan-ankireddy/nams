clc;
clearvars;
close all;

Alpha = 2:1:18;
alpha_list = 10.^(-Alpha/10);

BER = zeros(length(alpha_list),2);
for i = 1:length(alpha_list)
    
    alpha = alpha_list(i);
    [b1, b2] = run_scheme(alpha);
    
    BER(i,1) = b1;
    BER(i,2) = b2;
end
%%

Alpha = 2:1:18;
figure;
semilogy(Alpha, BER(:,1))
hold on;
semilogy(Alpha, BER(:,2))
grid on
xlabel('Attenuation')
ylabel('BER')
title('Theoretical')
legend('First Transmission','Last Transmission','location','best')

B = open('/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Datasets/Data_39542.mat');
Alpha = B.Alpha.';
Alpha = flipud(Alpha);
B = B.BER;
B1 = flipud(B(:,1));
B2 = flipud(B(:,2));

figure;
semilogy(Alpha, B1)
hold on;
semilogy(Alpha, B2)
grid on;
xlabel('Attenuation')
ylabel('BER')
title('OTA')
legend('First Transmission','Last Transmission','location','best')
