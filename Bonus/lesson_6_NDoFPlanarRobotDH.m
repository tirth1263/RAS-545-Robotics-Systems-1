classdef NDoFPlanarRobotDH
    % Minimal n-DoF planar revolute robot (unit links)
    properties
        n  % number of joints
        l  % link lengths (all ones)
    end
    methods
        function obj = NDoFPlanarRobotDH(n)
            obj.n = n;
            obj.l = ones(1,n);
        end

        % Forward kinematics
        function [p, psi] = fk(obj, q)
            csum = cumsum(q(:));
            x = sum(obj.l(:) .* cos(csum));
            y = sum(obj.l(:) .* sin(csum));
            p = [x; y];
            psi = sum(q(:));
        end

        % Translation Jacobian (2xN)
        function J = Jpos(obj, q)
            J = zeros(2, obj.n);
            csum = cumsum(q(:));
            for k = 1:obj.n
                sx = 0; sy = 0;
                for i = k:obj.n
                    sx = sx - obj.l(i)*sin(csum(i));
                    sy = sy + obj.l(i)*cos(csum(i));
                end
                J(:,k) = [sx; sy];
            end
        end

        % Orientation Jacobian
        function Jr = Jrot(obj, ~)
            Jr = ones(1, obj.n);
        end

        % Pose Jacobian
        function J = Jpose(obj, q)
            J = [obj.Jpos(q); obj.Jrot(q)];
        end

        % Simple plot
        function plot2d(obj, q, ttl)
            pts = zeros(2, obj.n+1);
            ang = 0;
            for i = 1:obj.n
                ang = ang + q(i);
                pts(:,i+1) = pts(:,i) + obj.l(i)*[cos(ang); sin(ang)];
            end
            plot(pts(1,:), pts(2,:), '-o', 'LineWidth', 2);
            axis equal; grid on; xlabel x; ylabel y;
            R = sum(obj.l)+0.5; xlim([-R R]); ylim([-0.5 R]);
            if nargin>2, title(ttl); end
            drawnow;
        end
    end
end
