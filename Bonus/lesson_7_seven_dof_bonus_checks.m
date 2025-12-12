% seven_dof_robot_bonus_checks.m
clear; close all; clc;
rng(0);

%% 1) Show A A^+ = U Σ V^T V Σ^† U^T = U Σ Σ^† U^T = I_{rank(A)}
m = 5; n = 8;
A = randn(m,n);
[U,S,V] = svd(A,'econ');           
Aplus = pinv(A);                    
P = A*Aplus;                        

% Compare to U*diag(1,..,1,0,..0)*U'
s = diag(S);
Sigma_pinv = zeros(size(S.'));      % n x m (just for shape understanding)
Sigma_pinv(1:numel(s), 1:numel(s)) = diag(1./s);
Irank = U(:,1:rank(A))*U(:,1:rank(A))';  % projector in U-space

fprintf('||A*A^+ - U_r*U_r^T|| = %.2e\n', norm(P - Irank));

%% 2) Show (B^T B + λ I) is invertible for any λ>0, even if rank(B)<m
m = 8; n = 5; lambda = 0.2;
B = randn(m,n);                      % rank(B) <= min(m,n)
M = (B.'*B + lambda*eye(n));         % SPD for lambda>0
fprintf('eig(M) min/max = [%.3f, %.3f]\n', min(eig(M)), max(eig(M)));
fprintf('cond(M) = %.2e\n', cond(M)); % finite -> invertible

