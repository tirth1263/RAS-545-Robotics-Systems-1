% r_perp_r_parallel_r_fk_compare_no_toolbox_safe.m
% Forward kinematics of an R ⟂ R ∥ R (3R) robot.
% (1) Try DQ Robotics toolbox. If not available, use a tiny pure-MATLAB
%     dual-quaternion fallback (no toolboxes, no function files).
% (2) Also compute FK with homogeneous transforms (manual DH).
% Compare position and orientation.

clear; clc; close all;

%% --- Robot geometry (standard DH, Craig) ---
% R ⟂ R ∥ R  =>  z0 ⟂ z1  and  z1 ∥ z2  --> α1 = +pi/2, α2 = 0
L1 = 0.30;   
L2 = 0.25;   
L3 = 0.15;  

% DH rows: [theta  d    a    alpha] 
DH = [ 0     L1   0.00   +pi/2;   
       0     0.0  L2      0.0 ;   
       0     0.0  L3      0.0 ];  

% Test joint configuration 
q = deg2rad([30; -20; 40]);   


%% --- (2) Manual homogeneous-transformation FK (standard DH) ---
A = @(th,d,a,al) [ cos(th) -sin(th)*cos(al)  sin(th)*sin(al)  a*cos(th); ...
                   sin(th)  cos(th)*cos(al) -cos(th)*sin(al)  a*sin(th); ...
                        0            sin(al)           cos(al)         d; ...
                        0                 0                 0          1 ];

th1 = DH(1,1) + q(1);  d1 = DH(1,2);  a1 = DH(1,3);  al1 = DH(1,4);
th2 = DH(2,1) + q(2);  d2 = DH(2,2);  a2 = DH(2,3);  al2 = DH(2,4);
th3 = DH(3,1) + q(3);  d3 = DH(3,2);  a3 = DH(3,3);  al3 = DH(3,4);

A1 = A(th1,d1,a1,al1);
A2 = A(th2,d2,a2,al2);
A3 = A(th3,d3,a3,al3);

T_ht = A1*A2*A3;
p_ht = T_ht(1:3,4);
R_ht = T_ht(1:3,1:3);


%% --- (1) DQ Robotics toolbox FK (with fallback that removes your error) ---
dq_toolbox_ok = true;
try
    robot_dq = DQ_SerialManipulatorDH(DH, 'standard'); 
catch
    dq_toolbox_ok = false;
end

if dq_toolbox_ok
    pose_dq = robot_dq.fkm(q);
    T_dq    = dq2ht(pose_dq);
else
    % ---------- Pure-MATLAB minimal DQ fallback (no toolboxes) ----------
    p = p_ht;    
    R = R_ht;

    % rotm -> quat (robust, no toolboxes)
    tr = trace(R);
    if tr > 0
        S = sqrt(tr+1.0)*2;
        qw = 0.25*S;
        qx = (R(3,2)-R(2,3))/S;
        qy = (R(1,3)-R(3,1))/S;
        qz = (R(2,1)-R(1,2))/S;
    else
        [~,mxi] = max([R(1,1), R(2,2), R(3,3)]);
        switch mxi
            case 1
                S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3))*2;
                qw = (R(3,2)-R(2,3))/S;
                qx = 0.25*S;
                qy = (R(1,2)+R(2,1))/S;
                qz = (R(1,3)+R(3,1))/S;
            case 2
                S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3))*2;
                qw = (R(1,3)-R(3,1))/S;
                qx = (R(1,2)+R(2,1))/S;
                qy = 0.25*S;
                qz = (R(2,3)+R(3,2))/S;
            otherwise
                S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2))*2;
                qw = (R(2,1)-R(1,2))/S;
                qx = (R(1,3)+R(3,1))/S;
                qy = (R(2,3)+R(3,2))/S;
                qz = 0.25*S;
        end
    end
    r = [qw qx qy qz]; r = r./norm(r);

    qmul  = @(a,b) [ a(1)*b(1) - dot(a(2:4),b(2:4)), ...
                     a(1)*b(2:4) + b(1)*a(2:4) + cross(a(2:4),b(2:4)) ];
    qconj = @(a) [a(1) -a(2) -a(3) -a(4)];

    d = 0.5 * qmul([1 0 0 0; 0 p.' ], r);  
    d = 0.5 * qmul([0 p(:).'], r);         

    % Reconstruct HT from dual quaternion:
    rc = qconj(r);
    tq = qmul( 2*d, rc );  t_rec = tq(2:4).';
    % quat -> rotm (explicit)
    w= r(1); x=r(2); y=r(3); z=r(4);
    R_rec = [1-2*(y^2+z^2),   2*(x*y - z*w),   2*(x*z + y*w);
             2*(x*y + z*w), 1-2*(x^2+z^2),    2*(y*z - x*w);
             2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x^2+y^2)];

    T_dq = eye(4); T_dq(1:3,1:3)=R_rec; T_dq(1:3,4)=t_rec;
end

p_dq = T_dq(1:3,4);
R_dq = T_dq(1:3,1:3);


%% --- Comparison: position and orientation
pos_err = norm(p_dq - p_ht, 2);

% Orientation error angle from relative rotation
R_rel = R_dq' * R_ht;
ang_err = acos( max(-1,min(1, (trace(R_rel)-1)/2 )) );  % radians

fprintf('--- R \\perp R \\parallel R FK comparison ---\n');
fprintf('q (deg): [%6.1f  %6.1f  %6.1f]\n', rad2deg(q));
fprintf('\nDQ pose (position):   [%.6f  %.6f  %.6f]^T\n', p_dq);
fprintf('HT pose (position):   [%.6f  %.6f  %.6f]^T\n', p_ht);
fprintf('\nPosition error ||p_dq - p_ht|| = %.3e m\n', pos_err);
fprintf('Orientation error angle         = %.3e rad (%.6f deg)\n', ang_err, rad2deg(ang_err));

%% --- (optional) show frames
figure; hold on; axis equal; grid on; view(3);
xlabel('X'); ylabel('Y'); zlabel('Z'); title('End-effector frames: DQ vs HT');

% draw triads
o1=T_dq(1:3,4); L=0.08;
quiver3(o1(1),o1(2),o1(3), L*T_dq(1,1), L*T_dq(2,1), L*T_dq(3,1), 0, 'LineWidth',2);
quiver3(o1(1),o1(2),o1(3), L*T_dq(1,2), L*T_dq(2,2), L*T_dq(3,2), 0, 'LineWidth',2);
quiver3(o1(1),o1(2),o1(3), L*T_dq(1,3), L*T_dq(2,3), L*T_dq(3,3), 0, 'LineWidth',2);
o2=T_ht(1:3,4);
quiver3(o2(1),o2(2),o2(3), L*T_ht(1,1), L*T_ht(2,1), L*T_ht(3,1), 0, 'LineWidth',2);
quiver3(o2(1),o2(2),o2(3), L*T_ht(1,2), L*T_ht(2,2), L*T_ht(3,2), 0, 'LineWidth',2);
quiver3(o2(1),o2(2),o2(3), L*T_ht(1,3), L*T_ht(2,3), L*T_ht(3,3), 0, 'LineWidth',2);
legend({'X_{DQ}','Y_{DQ}','Z_{DQ}','X_{HT}','Y_{HT}','Z_{HT}'}, 'Location','bestoutside');
