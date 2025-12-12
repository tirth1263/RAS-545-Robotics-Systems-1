% 1. Define real numbers
a1 = 123;
a2 = 321;
%%
% 2. Calculate and display a3 = a1 + a2
a3 = a1 + a2;
disp(['a1 + a2 = ', num2str(a3)]);
%%
% 3. Calculate and display a3 = a1 * a2
a3 = a1 * a2;
disp(['a1 * a2 = ', num2str(a3)]);
%%
% 4. Calculate and display a3 = a1 / a2
a3 = a1 / a2;
disp(['a1 / a2 = ', num2str(a3)]);
%%
% 5. Define theta1 and theta2
theta1 = pi/12;
theta2 = 3 * pi;
%%
% 6. Calculate and display a3 = cos(theta1 + theta2)
a3 = cos(theta1 + theta2);
disp(['cos(theta1 + theta2) = ', num2str(a3)]);
%%
% 7. Calculate and display a3 = cos^2(theta1 + theta2) + sin^2(theta1 + theta2)
a3 = cos(theta1 + theta2)^2 + sin(theta1 + theta2)^2;
disp(['cos^2 + sin^2 = ', num2str(a3)]);
%%
% 8. Calculate and display a3 = exp(a1) + log(a2)
a3 = exp(a1) + log(a2);
disp(['exp(a1) + log(a2) = ', num2str(a3)]);
%%
% Bonus: Display the base of natural log (e) with 15 significant digits
disp(['Value of e = ', num2str(exp(1), 15)]);
%%