h = 53.5;
L1 = 150;
L2 = 150;
offset = [0; 0; -58.5];

targets = [
 240, 0, 145;
 0, 271.20683206651610697117341781528, 60.95389311788625760811639159871;
 300, 50, 100;
 280, -195, 15;
 35, 270, -33;
];

maxIter = 100;
tol = 1e-3;
alpha = 0.5;
delta = 1e-6;

% Forward Kinematics (FK) function handle for cleaner main loop
FK = @(theta) compute_fk(theta, h, L1, L2, offset);

for i = 1:size(targets, 1)
    target = targets(i, :)';
    theta = [0.1; 0.1; 0.1]; % Initial guess
    
    for k = 1:maxIter
        pos = FK(theta);
        error = target - pos;
        
        if norm(error) < tol
            break;
        end
        
        J = zeros(3, 3);
        
        for j = 1:3
            theta_p = theta;
            theta_p(j) = theta_p(j) + delta;
            
            pos_p = FK(theta_p);
            
            % Numerical Jacobian column
            J(:, j) = (pos_p - pos) / delta;
        end
        
        % Update joint angles using Pseudo-inverse
        d_theta = alpha * pinv(J) * error;
        theta = theta + d_theta;
        
        % Normalize angles to [-pi, pi]
        theta = mod(theta + pi, 2*pi) - pi;
    end
    
    fprintf('Target %d joint angles (deg): t1=%.2f, t2=%.2f, t3=%.2f\n', i, theta(1)*180/pi, theta(2)*180/pi, theta(3)*180/pi);
end

% Helper Function: Forward Kinematics (FK)
function pos = compute_fk(theta, h, L1, L2, offset)
    t1 = theta(1);
    t2 = theta(2);
    t3 = theta(3);
    
    % H1: Base to Joint 1
    R1 = [cos(t1) -sin(t1) 0; sin(t1) cos(t1) 0; 0 0 1];
    M1 = [1 0 0; 0 0 1; 0 -1 0]; 
    H1 = [R1*M1, [0; 0; h]; 0 0 0 1];
    
    % H2: Joint 1 to Joint 2
    R2 = [cos(t2) -sin(t2) 0; sin(t2) cos(t2) 0; 0 0 1];
    M2 = [0 -1 0; 1 0 0; 0 0 1]; 
    H2 = [R2*M2, [90+L1*sin(t2); -L1*cos(t2); 0]; 0 0 0 1];
    
    % H3: Joint 2 to End-Effector
    R3 = [cos(t3-t2) -sin(t3-t2) 0; sin(t3-t2) cos(t3-t2) 0; 0 0 1];
    H3 = [R3, [L2*sin(t3-t2); -L2*cos(t3-t2); 0]; 0 0 0 1];
    
    % Total Transformation
    H = H1 * H2 * H3;
    
    % Extract position vector and apply offset
    pos = H(1:3, 4) + offset;
end