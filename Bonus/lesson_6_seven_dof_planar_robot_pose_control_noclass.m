% seven_dof_planar_robot_pose_control_noclass.m

clear; close all; clc;

n = 7; l = ones(n,1);
q = (pi/8)*ones(n,1);
u_t   = [2; 1];         
phi_d = -pi/4;          

fk = @(q,l) [ sum(l.*cos(cumsum(q)));
              sum(l.*sin(cumsum(q))) ];
psi = @(q) sum(q);

dt = 0.02; gain = 2.0; tol = 1e-4; max_steps = 6000;
traj = [];

figure; hold on; axis equal; grid on;

for k=1:max_steps
    p = fk(q,l);  ang = psi(q);
    e = [u_t - p; phi_d - ang];          

    traj(:,end+1) = p; 
    if norm(e) < tol, break; end

    % Build full pose Jacobian J = [Jpos; Jrot]
    csum = cumsum(q);
    Jpos = zeros(2,n);
    for i = 1:n
        Jpos(1,i) = -sum(l(i:end).*sin(csum(i:end)));
        Jpos(2,i) =  sum(l(i:end).*cos(csum(i:end)));
    end
    Jrot = ones(1,n);
    J = [Jpos; Jrot];                  % 3×n

    qdot = pinv(J) * (gain*e);
    q = q + dt*qdot;

    if mod(k,10)==0 || k==1
        clf; hold on; axis equal; grid on;
        pts=zeros(2,n+1); a=0;
        for i=1:n
            a=a+q(i);
            pts(:,i+1)=pts(:,i)+l(i)*[cos(a); sin(a)];
        end
        plot(pts(1,:),pts(2,:),'-o','LineWidth',2);
        R=sum(l)+0.5; xlim([-R R]); ylim([-0.5 R]);
        plot(traj(1,:),traj(2,:),'.','MarkerSize',10);
        plot(u_t(1),u_t(2),'rx','LineWidth',2,'MarkerSize',10);
        xlabel x; ylabel y; title('7-DoF planar: pose control');
        legend('links','traj','target','Location','best');
        drawnow;
    end
end

disp('Final q (rad):'); disp(q.');
p = fk(q,l); ang = psi(q);
disp('Final position [x y]:'); disp(p.');
disp('Final psi (rad):'); disp(ang);
