clear all;
close all;
clc;

addpath('..\')

a = [1 + 1*i  2   ];
b = [2 + 1*i  1+i ];

af = fixed_point(a,2,4);
bf = fixed_point(b,2,4);
cf = af./bf;
cf = transform(cf,3,4);




af.data
bf.data
cf.data


% c  = fixed_point(20.51321,6,3);
% d  = transform(c,3,0);
% c.to_digit
% d.to_digit