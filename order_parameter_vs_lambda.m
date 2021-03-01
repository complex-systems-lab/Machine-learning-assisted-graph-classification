global N omega a
N = 500;
tspan = 0:0.01:500;
vals = readmatrix('initial_values/initial_values_500.txt');
y0 = vals(:,1);
omega = readmatrix('initial_values/gaussian_freq.txt');
r_values = [];
lamdba_values = [];
a = load(sprintf('ER+SF_500_k10/er%i.txt',i));
lambda = 0;

for lambda = 0.1:0.01:0.3
    [t,y] = ode45(@(t,y)odefm(t,y,lambda),tspan,y0);
    y = wrapToPi(y);
    r = order_par(y(30000:50000,:));
    r_values = [r_values; r];
    lambda_values = [lambda_values; lambda];
    disp(i);
end
writematrix([lambda_values, r_values], 'order_parameter_vs_lambda.txt');
  
function theta_dot = odefm(~,theta,lambda)
     global omega a
     theta_dot = omega + lambda*sum(a.*sin(theta-theta'))';
end

function r=order_par(x)
    global N
    r1=abs((sum(exp(1i*x),2))/N);
    R=mean(r1);
    r= R;
end
