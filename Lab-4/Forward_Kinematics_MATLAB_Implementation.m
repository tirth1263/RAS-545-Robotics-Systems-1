% Define symbolic variables for the joint angles
syms t1 t2 t3

% --- Physical Parameters ---
h = 53.5;
L1 = 150;
L2 = 150;
offset = [0; 0; -54.5];

% --- Homogeneous Transformation Matrix H1 (Base to Joint 1) ---
R1 = [cos(t1) -sin(t1) 0;
      sin(t1) cos(t1) 0;
      0 0 1];
M1 = [1 0 0;
      0 0 1;
      0 -1 0];
T1 = [0; 0; h];
H1 = [R1 * M1, T1; 
      0 0 0, 1];

% --- Homogeneous Transformation Matrix H2 (Joint 1 to Joint 2) ---
R2 = [cos(t2) -sin(t2) 0;
      sin(t2) cos(t2) 0;
      0 0 1];
M2 = [0 -1 0;
      1 0 0;
      0 0 1];
T2 = [90+L1*sin(t2); -L1*cos(t2); 0];
H2 = [R2 * M2, T2; 
      0 0 0, 1];

% --- Homogeneous Transformation Matrix H3 (Joint 2 to End-Effector) ---
R3 = [cos(t3-t2) -sin(t3-t2) 0;
      sin(t3-t2) cos(t3-t2) 0;
      0 0 1];
M3 = [1 0 0;
      0 1 0;
      0 0 1];
T3 = [L2*sin(t3-t2); -L2*cos(t3-t2); 0];
H3 = [R3 * M3, T3; 
      0 0 0, 1];

% --- Symbolic Final Transformation Matrix ---
FinalH = simplify(H1 * H2 * H3);
fprintf('Symbolic Final Homogeneous Matrix (FinalH) Calculated.\n');

% --- Single Test Case ---
ang1 = 10;
ang2 = 20;
ang3 = 40;
r1 = deg2rad(ang1);
r2 = deg2rad(ang2);
r3 = deg2rad(ang3);

FinalH_num = subs(FinalH, [t1, t2, t3], [r1, r2, r3]);
pos = FinalH_num(1:3, 4);
final_pos = pos + offset;

fprintf('\n--- Single Test Case (10, 20, 40 deg) ---\n');
disp('End-effector position (x, y, z):');
vpa(final_pos)

% --- Multiple Test Cases ---
Theta = [
 0 0 0;
 90 20 30;
 -2 25 23;
 31 52 42;
 27 52 24
];

num_sets = size(Theta, 1);
fprintf('\n--- Multiple Test Cases (%d sets) ---\n', num_sets);

for i = 1:num_sets
    theta_deg = Theta(i, :);
    theta_rad = deg2rad(theta_deg);
    
    % Substitute numerical values into the symbolic matrix
    FinalH_num = subs(FinalH, [t1, t2, t3], theta_rad);
    
    % Extract position vector and apply offset
    pos = FinalH_num(1:3, 4);
    final_pos = pos + offset;
    
    disp(['Theta set ', num2str(i), ' (', num2str(theta_deg), ' deg):']);
    disp('Homogeneous matrix:');
    disp(vpa(FinalH_num));
    disp('Final position (x, y, z):');
    disp(vpa(final_pos));
end