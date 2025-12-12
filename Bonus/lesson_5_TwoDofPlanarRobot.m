clear; close all; clc;
%% 
%% Robot "model" (just store lengths)
l1 = 1; 
l2 = 1;

%% Target and initial state
q = [0; pi/2];           
p_target = [0; 2];       

%% Controller settings
clear; close all; clc;

% --- Robot parameters ---
l1 = 1; 
l2 = 1;
q = [0; pi/2];
p_target = [0; 2];

% --- Inline kinematics ---
fk = @(q,l1,l2) [l1*cos(q(1)) + l2*cos(q(1)+q(2));
                 l1*sin(q(1)) + l2*sin(q(1)+q(2))];

jacobian = @(q,l1,l2) [-l1*sin(q(1)) - l2*sin(q(1)+q(2)), -l2*sin(q(1)+q(2));
                        l1*cos(q(1)) + l2*cos(q(1)+q(2)),  l2*cos(q(1)+q(2))];

% --- Controller settings ---
dt   = 0.02; 
gain = 2.0; 
tol  = 1e-4; 
max_steps = 2000;
trajP = [];

figure; hold on; axis equal; grid on;

for k = 1:max_steps
    p = fk(q,l1,l2);
    e = p_target - p;
    trajP(:,end+1) = p; 
    if norm(e) < tol, break; end

    J = jacobian(q,l1,l2);
    qdot = pinv(J) * (gain * e);
    q = q + dt * qdot;

    if mod(k,10)==0 || k==1
        clf; hold on; axis equal; grid on;

        % --- inline plotting (replaces plot_links) ---
        th1 = q(1); th2 = q(2);
        p0 = [0;0];
        p1 = [l1*cos(th1);                 l1*sin(th1)];
        p2 = [l1*cos(th1)+l2*cos(th1+th2); l1*sin(th1)+l2*sin(th1+th2)];
        plot([p0(1) p1(1) p2(1)], [p0(2) p1(2) p2(2)], '-o', 'LineWidth', 2);

        R = l1 + l2 + 0.5;
        xlim([-R R]); ylim([-0.5 R]);
        plot(trajP(1,:), trajP(2,:), '.', 'MarkerSize', 10);
        plot(p_target(1), p_target(2), 'rx', 'LineWidth', 2, 'MarkerSize', 10);
        xlabel x; ylabel y; title('2-DoF planar robot (no class)');
        legend('links','trajectory','target','Location','best');
        drawnow;
    end
end

disp('Final joint angles [rad]:'); disp(q(:).');
disp('Final end-effector position:'); disp(fk(q,l1,l2).');

