% seven_dof_planar_robot_rotation_control_noclass.m

clear; close all; clc;

n = 7; l = ones(n,1);
q = (pi/8)*ones(n,1);

phi_des = -pi/4;                

% Orientation psi is sum of joint angles
psi = @(q) sum(q);

% Controller params
dt=0.02; gain=2.0; tol=1e-5; max_steps=4000;

figure; hold on; axis equal; grid on;

for k=1:max_steps
    e = phi_des - psi(q);        
    if abs(e) < tol, break; end

    Jr = ones(1,n);              
    qdot = (Jr'/(Jr*Jr')) * (gain*e);   
    q = q + dt*qdot;

    if mod(k,20)==0 || k==1
        clf; hold on; axis equal; grid on;
        % inline plotting
        pts=zeros(2,n+1); ang=0;
        for i=1:n
            ang=ang+q(i);
            pts(:,i+1)=pts(:,i)+l(i)*[cos(ang); sin(ang)];
        end
        plot(pts(1,:),pts(2,:),'-o','LineWidth',2);
        R=sum(l)+0.5; xlim([-R R]); ylim([-0.5 R]);
        xlabel x; ylabel y;
        title(sprintf('Rotation control: psi=%.3f rad', psi(q)));
        drawnow;
    end
end

disp('Final q (rad):'); disp(q.');
disp('Final psi (rad):'); disp(psi(q));
