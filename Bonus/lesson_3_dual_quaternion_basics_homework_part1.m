% dual_quaternion_basics_homework_part1.m

clear; close all; clc;
%% %% --------- Minimal quaternion & dual quaternion helpers ---------
% quaternion as [w x y z]
qmul  = @(q2,q1) [ ...
    q2(1)*q1(1) - dot(q2(2:4),q1(2:4)), ...
    q2(1)*q1(2:4) + q1(1)*q2(2:4) + cross(q2(2:4),q1(2:4)) ];
qconj = @(q) [q(1) -q(2) -q(3) -q(4)];
qrot  = @(axis,phi) [cos(phi/2), sin(phi/2)*axis(:)'./(norm(axis)+eps)];
%% % dual quaternion as pair (r,d) where X = r + ε d  (store as 1x8: [r d])
dq    = @(r,d) [r d];
dql   = @(X) X(1:4);             % primal (rotation quaternion)
dqd   = @(X) X(5:8);             % dual part
dqmul = @(X2,X1) dq( qmul(dql(X2),dql(X1)), ...
                     qmul(dql(X2),dqd(X1)) + qmul(dqd(X2),dql(X1)) );
dqconj= @(X) dq( qconj(dql(X)), qconj(dqd(X)) );
%% 
% build pose from rotation r and translation t (1x3)
dq_from_rt = @(r,t) dq( r, 0.5*qmul([0 t], r) );

%% % extract translation vector from unit dual quaternion X (no invalid indexing)
vec = @(qq) qq(2:4);                                 % helper: take vector part
dq_t = @(X) vec( 2 * qmul( dqd(X), qconj( dql(X) ) ) );
%% % rotation matrix from quaternion r
q2R = @(r) [ ...
  1-2*(r(3)^2+r(4)^2), 2*(r(2)*r(3)-r(1)*r(4)), 2*(r(2)*r(4)+r(1)*r(3));
  2*(r(2)*r(3)+r(1)*r(4)), 1-2*(r(2)^2+r(4)^2), 2*(r(3)*r(4)-r(1)*r(2));
  2*(r(2)*r(4)-r(1)*r(3)), 2*(r(3)*r(4)+r(1)*r(2)), 1-2*(r(2)^2+r(3)^2) ];

%% %% --------- Homework ---------
%% % x0 = identity
x0 = dq([1 0 0 0],[0 0 0 0]);
%% % 1) x1: translation +1 m along x, then rotation +pi/8 about z
Rt1 = qrot([0 0 1], pi/8);                 % rotation about z
T1  = [1 0 0];                              % translation 1 along x
Xtr = dq_from_rt([1 0 0 0], T1);            % pure translation
Xrot= dq_from_rt(Rt1, [0 0 0]);             % pure rotation
x1  = dqmul(Xrot, Xtr);                     % apply translation, then rotation
%% % 2) x2: translation -1 m along z, then rotation -pi/2 about x
Rt2 = qrot([1 0 0], -pi/2);
T2  = [0 0 -1];
x2  = dqmul( dq_from_rt(Rt2,[0 0 0]), dq_from_rt([1 0 0 0], T2) );
%% % 3) sequential transform of neutral frame by x0, then x1, then x2
x3 = dqmul(x2, dqmul(x1,x0));
%% % 4) rotation of x3 without using rotation(.)
r3 = dql(x3);   % quaternion part
disp('Rotation quaternion of x3:');
disp(r3);
%% % 5) translation of x3 without using translation(.)
t3 = dq_t(x3);
disp('Translation vector of x3 [tx ty tz]:');
disp(t3);
%%  % 6) x4: rotation -pi/8 about z, followed by translation +1 m in x
x4 = dqmul( dq_from_rt([1 0 0 0], [1 0 0]), dq_from_rt(qrot([0 0 1], -pi/8), [0 0 0]) );
disp('Is x4 equal to x1? (should be false because order matters)');
disp(norm(x4 - x1) < 1e-12);
%% 
%% 
%% 
%% 