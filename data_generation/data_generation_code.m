global N omega a
N = 500;  %the size of the network being used
tspan = 0:0.01:500; %the total time for which the simulation is being run

vals = readmatrix('_file_containing_initial_phases_'); 
y0 = vals(:,1); % read the first column containing phase values,taken same for all graphs
omega = readmatrix('_file_containing_gaussian_omega_values');

lamda = #; % lambda value for the simulation

% loop to generate data for 250 different graphs of one class for which the adjacency matrix is being input
for i = 1:250
    % omega = readmatrix(sprintf('_file_containing_ith_omega_values_', i)); % run only if different frequency values are to be used for each graph
    a = readmatrix('_file_containing_ith_adjacency_matrix_'); % read the adjacency matrix of each graph
    k = sum(a);
    [m,sortorder] = sort(k,'descend'); % find the descending order of degree
    [t,y] = ode45(@(t,y)odefm(t,y,lamda), tspan, y0); % the inbuilt function to apply the 4th order Runge-Kutta method
    disp(size(y));
    phases = [];
    y = wrapToPi(y);
    y = y(:, sortorder); % arrange the phase values according to the descending order of degree or corresponding nodes
    % rand = randi(491); % to be used if random nodes are required
    phases = [phases, y(30000:50000, 1:10)]; % leaving the initial transient of 30000, 20000 phase values are recorded for top ten nodes
    temp = pp(2:100:20001, :);
    phases = phases(1:100:20000, :);
    c = phases < temp; % the phase values are converted to symbolic values
    writematrix(c,'_filename_to_save_ith_output_data_'); % output will of the shape (200,10)
end

%function called by ode45 to generate consecutive theta values
function theta_dot = odefm(~,theta,lamda)
     global omega a
     theta_dot = omega + lamda*sum(a.*sin(theta-theta'))'; % the theta values for the ode45 function are calculated using this equation
end

% function to generate random values in range -val to val
function random = generate_random(val, N)
    random = -val + rand(1,N)*2*val;
end
