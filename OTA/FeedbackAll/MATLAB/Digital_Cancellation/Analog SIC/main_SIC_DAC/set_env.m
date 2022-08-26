function set_env()

addpath('../OFDM')
addpath('../rf_to_bsb')
addpath('../chest')
addpath('../wrapper')
addpath(genpath('../SIC'))
addpath('../eth')
addpath('../utilities')
addpath(genpath('../TTD_model/'))
addpath('../SIC_DAC/')

randn('state',1234)
rand('state',12345)
