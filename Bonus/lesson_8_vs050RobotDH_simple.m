% VS050RobotDH_simple.m  

% ---- DH table: [theta  d   a   alpha] ----
DH_TABLE = [ ...
    0      0.345   0.000   +pi/2;  % J1
    0      0.000   0.250    0.0 ;  % J2
    0      0.000   0.160    0.0 ;  % J3
    0      0.250   0.000   +pi/2;  % J4
    0      0.000   0.000   -pi/2;  % J5
    0      0.100   0.000    0.0 ]; % J6

shaft_len = 0.20;   % 20 cm shaft along +z6

% Build robot struct with function handles (helpers are separate files)
r = struct();
r.DH        = DH_TABLE;
r.shaft_len = shaft_len;
r.fk   = @(q) fk_vs050(q, r.DH, r.shaft_len);
r.Jpos = @(q) numJ_pos(@(qq) fk_vs050(qq, r.DH, r.shaft_len), q);
r.plot = @(q,ttl) plot_vs050(q, r.DH, r.shaft_len, ttl);

% Example use (optional):
% q = zeros(6,1); [T,p] = r.fk(q), J = r.Jpos(q); r.plot(q,'VS050')
