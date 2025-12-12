% fk_vs050_calc.m  

if ~exist('DH','var')
    % DH rows: [theta  d     a     alpha]
    DH = [ 0  0.345  0.000  +pi/2;   % J1
           0  0.000  0.250   0.0 ;   % J2
           0  0.000  0.160   0.0 ;   % J3
           0  0.250  0.000  +pi/2;   % J4
           0  0.000  0.000  -pi/2;   % J5
           0  0.100  0.000   0.0 ];  % J6
end
if ~exist('q','var'),         q = zeros(6,1);     end
if ~exist('shaft_len','var'), shaft_len = 0.20;   end

% Forward kinematics
T = eye(4);
for i = 1:6
    th = DH(i,1) + q(i);
    d  = DH(i,2);
    a  = DH(i,3);
    al = DH(i,4);
    A = [cos(th) -sin(th)*cos(al)  sin(th)*sin(al) a*cos(th); ...
         sin(th)  cos(th)*cos(al) -cos(th)*sin(al) a*sin(th); ...
              0            sin(al)           cos(al)          d; ...
              0                 0                 0           1];
    T = T * A;
end

T06 = T;
p_tool = T06(1:3,1:3) * [0;0;shaft_len] + T06(1:3,4);

