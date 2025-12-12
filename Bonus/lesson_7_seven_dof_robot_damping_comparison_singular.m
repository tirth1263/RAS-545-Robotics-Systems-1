% seven_dof_robot_damping_comparison_singular.m
clear; close all; clc;

% Robot (7-DoF planar, unit links)
n  = 7;               
l  = ones(n,1);       
q0 = zeros(n,1);      
ut = [2; 1];          

% Control params
eta = 10;             
tau = 0.01;          
T   = 4.0;            
K   = round(T/tau); 

% Forward kinematics (anonymous)
fk = @(q) [ sum(l.*cos(cumsum(q))); sum(l.*sin(cumsum(q))) ];

% Damping set (8 values)
lset = [0.001 0.01 0.1 0.2 0.3 0.4 0.5 1.0];

% Containers
errs  = cell(1, 1+numel(lset));   
uNorm = cell(1, 1+numel(lset));   
names = cell(1, 1+numel(lset));

% ---------- Controller A: SVD pseudoinverse ----------
q = q0;
en = zeros(1,K); un = zeros(1,K);
for k = 1:K
    % FK and error
    p = fk(q); e = ut - p; en(k) = norm(e);
    % Build Jacobian J_pos (2xN)
    csum = cumsum(q); J = zeros(2,n);
    for i = 1:n
        J(1,i) = -sum(l(i:end).*sin(csum(i:end)));
        J(2,i) =  sum(l(i:end).*cos(csum(i:end)));
    end
    % SVD pseudoinverse step
    qdot = pinv(J) * (eta*e);
    un(k) = norm(qdot);
    q = q + tau*qdot;
end
errs{1}  = en; uNorm{1} = un; names{1} = 'pinv (SVD)';

% ---------- Controllers B: damped least-squares ----------
for j = 1:numel(lset)
    lam = lset(j);
    q = q0; en = zeros(1,K); un = zeros(1,K);
    for k = 1:K
        p = fk(q); e = ut - p; en(k) = norm(e);
        csum = cumsum(q); J = zeros(2,n);
        for i = 1:n
            J(1,i) = -sum(l(i:end).*sin(csum(i:end)));
            J(2,i) =  sum(l(i:end).*cos(csum(i:end)));
        end
        % Damped pseudoinverse: J^T (J J^T + λ^2 I)^(-1)
        A = J*J.' + (lam^2)*eye(2);
        qdot = J.'*(A \ (eta*e));
        un(k) = norm(qdot);
        q = q + tau*qdot;
    end
    errs{1+j}  = en;
    uNorm{1+j} = un;
    names{1+j} = sprintf('DLS \\lambda=%.3g', lam);
end

% ---------- Plots ----------
mxK = max(cellfun(@numel, errs));
subplot(1,2,1); hold on; grid on;
for i = 1:numel(errs)
    plot((0:mxK-1)*tau, errs{i}, 'LineWidth', 1.5);
end
xlabel('time [s]'); ylabel('||e_t||'); title('Task-space error norm');
legend(names, 'Location','northeastoutside');

subplot(1,2,2); hold on; grid on;
for i = 1:numel(uNorm)
    plot((0:mxK-1)*tau, uNorm{i}, 'LineWidth', 1.5);
end
xlabel('time [s]'); ylabel('||\Delta q / \Delta t||'); title('Control signal norm');
legend(names, 'Location','northeastoutside');