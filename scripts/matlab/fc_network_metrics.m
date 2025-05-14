%fc_network_metrics.m v1.0
%The objective of this code generate network metrics from connecomes using
%Brain Connectivity Toolbox

%This code can be used after fc_conn_2_txt.m or connecomte_dwi.sh

%The input files are: SC or FC connecotmes
%The output files is: degree, cluster or betweeness centrality metrics

%Paths
source_dir = 'results/fc_dk/weight';
output_dir = 'results/fc_dk/weight/network';

% Initialize matrices to hold the results
degree_matrix = zeros(92, 84);
cluster_matrix = zeros(92, 84);
betw_matrix = zeros(92, 84);

% Not all subjects will have a FC connecotme.
matrices = cell(1, 92);

% Loop through the .txt files, read them, and compute metrics
for i = 1:100

    % Paths
    filename = sprintf('sub-%02d_fcw_connectome.txt', i);
    txt_file_path = fullfile(source_dir, filename);
    
    % Check if the .txt file exists
    if exist(txt_file_path, 'file')

        % Read the matrix, converts NaN values
        matrices{i} = readmatrix(txt_file_path);
        matrices{i}(isnan(matrices{i})) = 0;
        
        % Display 
        disp(['Processing Subject : ', num2str(i)]);
        
        % Compute metrics
        degree_matrix(i, :) = degrees_und(matrices{i});
        cluster_matrix(i, :) = clustering_coef_wu(matrices{i})';
        betw_matrix(i, :) = betweenness_wei(matrices{i})';

    else
        disp(['File does not exist: ', txt_file_path]);
    end
end

% Save the matrices to .txt files in the specified path with comma as delimiter
writematrix(degree_matrix, fullfile(output_dir, 'degree_w.txt'), 'Delimiter', 'comma');
writematrix(cluster_matrix, fullfile(output_dir, 'cluster_w.txt'), 'Delimiter', 'comma');
writematrix(betw_matrix, fullfile(output_dir, 'betw_w.txt'), 'Delimiter', 'comma');

disp('All metrics successfully saved.');

%% Network metrics for binarized fcb
%Same as previous, but for binarized
source_dir = 'results/fc_dk/binary';
output_dir = 'results/fc_dk/binary/network';


degree_matrix = zeros(92, 84);
cluster_matrix = zeros(92, 84);
betw_matrix = zeros(92, 84);
matrices = cell(1, 92);

for i = 1:100
    
    filename = sprintf('sub-%02d_fcb_connectome.txt', i);
    txt_file_path = fullfile(source_dir, filename);
    

    if exist(txt_file_path, 'file')


        matrices{i} = readmatrix(txt_file_path);
        matrices{i}(isnan(matrices{i})) = 0;
        
        disp(['Processing Subject : ', num2str(i)]);
        
        degree_matrix(i, :) = degrees_und(matrices{i});
        cluster_matrix(i, :) = clustering_coef_bu(matrices{i})';
        betw_matrix(i, :) = betweenness_bin(matrices{i})';

    else
        disp(['File does not exist: ', txt_file_path]);
    end
end

% Save the matrices to .txt files in the specified path with comma as delimiter
writematrix(degree_matrix, fullfile(output_dir, 'degree_b.txt'), 'Delimiter', 'comma');
writematrix(cluster_matrix, fullfile(output_dir, 'cluster_b.txt'), 'Delimiter', 'comma');
writematrix(betw_matrix, fullfile(output_dir, 'betw_b.txt'), 'Delimiter', 'comma');

disp('All metrics successfully saved.');

%--------------------------------------------------------------------------------------
%Version v1.0.
%--------------------------------------------------------------------------------------
%Get the lastest version at:
%--------------------------------------------------------------------------------------
%script by Alejandro Garma Oehmichen
%--------------------------------------------------------------------------------------
