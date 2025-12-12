% quaternion_basics_homework.m

clear; close all; clc;
%%
%% Minimal quaternion helpers (Hamilton convention, q = [w x y z])
q = @(w,x,y,z) [w x y z];
%%
% unit rotation quaternion from axis (ux,uy,uz) and angle phi (rad)
qrot = @(ux,uy,uz,phi) ...
    [cos(phi/2), sin(phi/2)*[ux uy uz]];
%%
% Hamilton product: q3 = q2 * q1  (apply q1 then q2)
qmul = @(q2,q1) [ ...
    q2(1)*q1(1) - dot(q2(2:4),q1(2:4)), ...
    q2(1)*q1(2:4) + q1(1)*q2(2:4) + cross(q2(2:4),q1(2:4)) ];
%%
% conjugate / inverse (for unit quaternions)
qconj = @(qq) [qq(1) -qq(2) -qq(3) -qq(4)];
%% 2) r1: rotation phi = pi/8 about x-axis
r1 = qrot(1,0,0, pi/8);
%% 3) r2: rotation phi = pi/16 about y-axis
r2 = qrot(0,1,0, pi/16);
%% 4) r3: rotation phi = pi/32 about z-axis
r3 = qrot(0,0,1, pi/32);
%% 5) Sequential rotation of the neutral frame by r1, then r2, then r3.
% Using Hamilton convention, "apply r1 then r2 then r3" => r4 = r3*r2*r1
r4 = qmul( qmul(r3,r2), r1 );
disp('r4 = r3 * r2 * r1 ='); disp(r4);
%% % 6) Reverse rotation of r4 (for unit q, inverse = conjugate)
r5 = qconj(r4);
disp('r5 (reverse of r4) ='); disp(r5);
%% % 7) Rotate r5 by 360° about x-axis → r6.  (2π about any axis = [-1 0 0 0])
qx_360 = qrot(1,0,0, 2*pi);
r6 = qmul(qx_360, r5);   % multiply on the left (apply 360° after r5)
disp('r6 = (360° about x) ∘ r5 ='); disp(r6);
%% % Check r5 == -r6 (same physical rotation)
disp(['Is r5 = -r6 ? ', string(norm(r5 + r6) < 1e-12)]);
%% 
%% 
%% %% ---------------- BONUS HOMEWORK ----------------

% 1) General quaternion multiplication:
%    h1 = a1 + b1 i + c1 j + d1 k,  h2 = a2 + b2 i + c2 j + d2 k
a1=0; b1=0; c1=0; d1=0;  % placeholders (set anything you like)
a2=0; b2=0; c2=0; d2=0;
%% % To show the general FORM, compute with symbols replaced by variables:
syms A1 B1 C1 D1 A2 B2 C2 D2 real
H1 = [A1 B1 C1 D1]; H2 = [A2 B2 C2 D2];
H3 = qmul(H2,H1);   % result of multiplying H2 * H1
% Components of h3 = a3 + b3 i + c3 j + d3 k:
a3 = H3(1); b3 = H3(2); c3 = H3(3); d3 = H3(4);
disp('General quaternion product (Hamilton):');
disp(['a3 = A2*A1 - (B2*B1 + C2*C1 + D2*D1)']);
disp(['b3 = A2*B1 + A1*B2 + (C2*D1 - D2*C1)']);
disp(['c3 = A2*C1 + A1*C2 + (D2*B1 - B2*D1)']);
disp(['d3 = A2*D1 + A1*D2 + (B2*C1 - C2*B1)']);

%% % 2) General form of quaternion norm:
%    ||h|| = sqrt(a^2 + b^2 + c^2 + d^2) = sqrt(h * h^*)
qnorm = @(qq) sqrt(sum(qq.^2));
disp('Quaternion norm formula:  ||h|| = sqrt(a^2 + b^2 + c^2 + d^2)');
%% % 3) Show r = cos(phi/2) + v*sin(phi/2) (unit v) has unit norm (numeric check)
phi = 1.2345;           % any angle
v = [1 2 3]; v = v/norm(v);
r = [cos(phi/2), sin(phi/2)*v];
disp(['Unit-check for r: ||r|| = ', num2str(qnorm(r), '%.12f')]);  % ≈ 1