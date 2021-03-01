N = 500; % the size of the network, i.e., the number of nodes in the network
initial_phases = generate_random(pi, N);

writematrix(initial_phases,'initial_phase_values.txt'); % write the phase values to a file named 'initial_phase_values.txt'

function random = generate_random(val, N)
    random = -val + rand(1,N)*2*val;
end
