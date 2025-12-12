% vs050_plot_simple.m

clear; close all; clc;

% --- Define DH and configuration ---
% DH rows: [theta  d      a      alpha]
DH = [ 0   0.345   0.000   +pi/2;   % J1
       0   0.000   0.250    0.0 ;   % J2
       0   0.000   0.160    0.0 ;   % J3
       0   0.250   0.000   +pi/2;   % J4
       0   0.000   0.000   -pi/2;   % J5
       0   0.100   0.000    0.0  ]; % J6
q = zeros(6,1);         
shaft_len = 0.20;      

% --- Forward chain to collect joint positions ---
T = eye(4);
P = zeros(3,7);
P(:,1) = [0;0;0];

for i = 1:6
    th = DH(i,1) + q(i);
    d  = DH(i,2);  a = DH(i,3);  al = DH(i,4);
    A = [cos(th) -sin(th)*cos(al)  sin(th)*sin(al) a*cos(th); ...
         sin(th)  cos(th)*cos(al) -cos(th)*sin(al) a*sin(th); ...
              0            sin(al)           cos(al)          d; ...
              0                 0                 0           1];
    T = T * A;
    P(:,i+1) = T(1:3,4);
end

% Tooltip (shaft tip) position
p_tool = T(1:3,1:3) * [0;0;shaft_len] + T(1:3,4);

% --- Plot ---
figure; hold on; grid on; axis equal; view(3);
plot3(P(1,:), P(2,:), P(3,:), '-o', 'LineWidth', 2);                       
plot3([P(1,7) p_tool(1)], [P(2,7) p_tool(2)], [P(3,7) p_tool(3)], 'r-', 'LineWidth', 2); 
xlabel('x'); ylabel('y'); zlabel('z'); title('VS050 simple plot');
