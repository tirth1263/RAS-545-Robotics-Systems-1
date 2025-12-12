% seven_dof_planar_robot_translation_control_noclass_fixed.m

clear; close all; clc;

% Robot definition
n  = 7;
l  = ones(n,1);                 
q  = (pi/8)*ones(n,1);          
u_t = [2; 1];                  

% Forward kinematics (simple, class-free)
fk = @(q,l) [ sum(l.*cos(cumsum(q)));
              sum(l.*sin(cumsum(q))) ];

% Controller params
dt = 0.02; gain = 2.0; tol = 1e-4; max_steps = 6000;
traj = [];

figure; hold on; axis equal; grid on;

for k = 1:max_steps
    p = fk(q,l);
    e = u_t - p;
    traj(:,end+1) = p; 
    if norm(e) < tol, break; end

    % Jacobian (2×n)
    csum = cumsum(q);
    J = zeros(2,n);
    for i = 1:n
        J(1,i) = -sum(l(i:end).*sin(csum(i:end)));
        J(2,i) =  sum(l(i:end).*cos(csum(i:end)));
    end

    % Resolved-rate update
    qdot = pinv(J) * (gain*e);
    q = q + dt*qdot;

    % Plot
    if mod(k,10)==0 || k==1
        clf; hold on; axis equal; grid on;
        pts = zeros(2,n+1); ang = 0;
        for i=1:n
            ang = ang + q(i);
            pts(:,i+1) = pts(:,i) + l(i)*[cos(ang); sin(ang)];
        end
        plot(pts(1,:),pts(2,:),'-o','LineWidth',2);
        R = sum(l)+0.5; xlim([-R R]); ylim([-0.5 R]);
        plot(traj(1,:),traj(2,:),'.','MarkerSize',10);
        plot(u_t(1),u_t(2),'rx','LineWidth',2,'MarkerSize',10);
        xlabel x; ylabel y; title('7-DoF planar: translation control');
        legend('links','trajectory','target','Location','best'); drawnow;
    end
end

disp('Final q (rad):'); disp(q.');
p = fk(q,l);
disp('Final position [x y]:'); disp(p.');
