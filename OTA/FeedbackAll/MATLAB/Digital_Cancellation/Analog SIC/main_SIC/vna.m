function app = vna()
% VNA
%   Starts the CMT VNA application with desired settings
%
% Args:
%   none
%
% Returns:
%   none

instrument = 'S2VNA';

use_center_and_span = false;

f1_hz           = 100e3;
f2_hz           = 819.2e6;
num_points      = 256;
if_bw           = 100;
power_level_dbm = -20; 

% prompt for measurement parameter
% default to S11
parameter = 'S21';

% prompt for format
% default to Log magnitude
format = 'LOGarithmic';

try
    app=actxserver([instrument,'.application']);
catch ME
    disp('Error establishing COM server connection.');
    disp('Check that the VNA application COM server was registered');
    disp('at the time of software installation.');
    disp('This is described in the VNA programming manual.');
    return
end

%Wait up to 20 seconds for instrument to be ready
ready = 0;
count = 0;
while ~ready
    ready = app.ready;
    if count > 20
        disp('Error, instrument not ready.');
        disp('Check that VNA is powered on and connected to PC.');
        disp('The status Ready should appear in the lower right');
        disp('corner of the VNA application window.');
        return
    end
    
    % Check every so often whether the instrument is ready yet
    pause(1)
    count = count + 1;
end

%Get and echo the instrument name, serial number, etc.
%
%  [This is a simple example of getting a property in MATLAB.]
%
disp(sprintf(app.name));

%Set the instrument to a Preset state
%
%  [This is an example of executing an ActiveX "method" in MATLAB.]
%
% app.scpi.system.invoke('preset');

%Configure the stimulus
if use_center_and_span
    app.scpi.get('sense',1).frequency.set('center',f1_hz);
    app.scpi.get('sense',1).frequency.set('span',f2_hz);
else
    
    app.scpi.get('sense',1).frequency.set('start',f1_hz);
    app.scpi.get('sense',1).frequency.set('stop',f2_hz);
end

app.scpi.get('sense',1).sweep.set('points',num_points);

if(instrument(1) ~= 'R')
    app.SCPI.get('SOURce',1).POWer.LEVel.IMMediate.set('AMPLitude',power_level_dbm);
end

app.SCPI.get('SENSE',1).BANDwidth.RESolution = if_bw;

app.scpi.get('calculate',1).get('parameter',1).set('define',parameter);

app.SCPI.TRIGger.SEQuence.SOURce = 'INT';

