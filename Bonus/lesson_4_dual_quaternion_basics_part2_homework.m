% dual_quaternion_basics_part2_homework.m

clear; close all; clc;
try, include_namespace_dq; end
%% %% Helpers
mk_line = @(p0,d) struct('d', d/norm(d), 'm', cross(p0, d/norm(d))); 
dist_point_line = @(p,L) norm(cross(L.d, p) - L.m);                 
mk_plane = @(n,d) struct('n', n(:)./norm(n), 'd', d);                
signed_dist_plane = @(p,Pi) (Pi.n(:).' * p(:)) - Pi.d;  
%% %% 1) Plücker line for the y-axis using 4 different points on it
d_y = [0 1 0];                 % direction of y-axis (unit)
pA = [0 0 0]; pB = [0 1 0]; pC = [0 -2 0]; pD = [0 3.5 0];
L1 = mk_line(pA, d_y);
L2 = mk_line(pB, d_y);
L3 = mk_line(pC, d_y);
L4 = mk_line(pD, d_y);
%% %% 2) Point p1 and its distance to L1
p1 = [2 1 1];
d_p1_L1 = dist_point_line(p1, L1);
%% %% 3) yz-plane using the two possible normals (±x̂)
pi1 = mk_plane([ 1 0 0], 0);  
pi2 = mk_plane([-1 0 0], 0);
%% %% 4) Signed distances from p1 to pi1 and pi2
d_p1_pi1 = signed_dist_plane(p1, pi1);   
d_p1_pi2 = signed_dist_plane(p1, pi2);
%% %% 5) Build a 5×5×5 m cube centered at origin, normals pointing inward
a = 2.5;  % half-length
pixm = mk_plane([ 1 0 0], -a);   
pixp = mk_plane([-1 0 0], -a);   
piym = mk_plane([ 0 1 0], -a);   
piyp = mk_plane([ 0 -1 0], -a); 
pizm = mk_plane([ 0 0 1], -a);  
pizp = mk_plane([ 0 0 -1], -a);  
%% % Plot cube edges, the y-axis line, and point p1
C = a*[-1 -1 -1;  1 -1 -1;  1  1 -1; -1  1 -1; -1 -1  1;  1 -1  1;  1  1  1; -1  1  1];
E = [1 2;2 3;3 4;4 1; 5 6;6 7;7 8;8 5; 1 5;2 6;3 7;4 8];
figure; hold on; axis equal; grid on; view(3);
for k=1:size(E,1)
    plot3(C(E(k,:),1),C(E(k,:),2),C(E(k,:),3),'k-','LineWidth',1.5);
end
%% % y-axis
plot3([0 0],[ -a a ],[0 0],'b--','LineWidth',1);
% point p1
plot3(p1(1),p1(2),p1(3),'ro','MarkerFaceColor','r');
xlabel X; ylabel Y; zlabel Z; title('5m Cube (normals inward), y-axis, and point p_1');
%% %% 6) Which cube plane is closest to p1?  What is that distance?
names = {'\pi_{x-}','\pi_{x+}','\pi_{y-}','\pi_{y+}','\pi_{z-}','\pi_{z+}'};
planes = {pixm, pixp, piym, piyp, pizm, pizp};

signed_d = cellfun(@(Pi) signed_dist_plane(p1,Pi), planes);
abs_d = abs(signed_d);
[closest_dist, idx] = min(abs_d);
closest_plane_name = names{idx};
%% % Display results
disp('--- Results ---');
disp('L1, L2, L3, L4 (Plücker lines for y-axis):');  % show (d,m)
disp([L1.d L1.m; L2.d L2.m; L3.d L3.m; L4.d L4.m]); % rows: [d m]
fprintf('Distance from p1 to L1: %.6f m\n', d_p1_L1);
fprintf('Signed distance p1 to pi1 ( +x normal ): %.6f m\n', d_p1_pi1);
fprintf('Signed distance p1 to pi2 ( -x normal ): %.6f m\n', d_p1_pi2);
fprintf('Closest cube plane to p1: %s\n', closest_plane_name);
fprintf('Distance from p1 to closest plane: %.6f m\n', closest_dist);