function TX_Feedback_Noiseless(num)
clc;

if nargin < 1
    num = 1;
end
currentFolder = pwd;
addpath(strcat(currentFolder, '/Imp_Files'));
addpath(strcat(currentFolder, '/Imp_Functions'));
run('Parameters_feedback.m');


B_Output = open(strcat('Feedback_Files/B',num2str(num),'_Output.mat'));
B_Output = B_Output.B_Output;

ZL = B_Output;
save(strcat('Feedback_Files/Z',num2str(num),'_Output.mat'),'ZL');
end