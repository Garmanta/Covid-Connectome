%fc_conn_2_txt.m v1.0
%The objective of this code extract the fc connectomes after processing
%from CONN, then to transform them to .txt files to be further processed in
%another code.

%Therefore, the use case is:
%CONN --> fc_conn_2_txt.m --> nbsfiltering.py

%This code does the following steps: 
% 1.- Open the CONN FC connecomtes
% 2.- Perform inverse fisher r to z transform
% 3.- Select the appropiate DK ROIs
% 4.- Save the CONN FC connectomes in a .txt

%The input files are: CONN FC connectomes
%The output files is: FC connectomes in .txt

% Paths
source_dir = 'conn_complete/conn_project01/results/firstlevel/RRC_dk';
output_dir = 'results/fc_dk';

% Ensure the output directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Loops through all the subject files. In this case, across 100 subjects
for i = 1:100
    
    %Path to the CONN FC connectome
    filename = sprintf('resultsROI_Subject%03d_Condition%03d.mat', i, 1);  
    mat_file_path = fullfile(source_dir, filename);
    
    % Load the Z matrix from the .mat file
    data = load(mat_file_path, 'Z');
    Z = data.Z;
    
    % CONN stores the connecotmes in z values. FC connectomes are usually reported as pearson r correlation values
    % To transform r to z values, one has to do an invewrse fisher r to z transform
    % This is just the tanh(Z values)
    R = tanh(Z);
    
    % Clip the matrix to the lower left section. Thank the default atlas in
    % CONN
    clipped_R = R(33:end, 33:end);
    
    % Output path
    txt_filename = sprintf('sub-%02d_fcw_connectome.txt', i);
    txt_file_path = fullfile(output_dir, txt_filename);
    
    % Save the FC connectome
    writematrix(clipped_R, txt_file_path, 'Delimiter', 'tab');
end

disp('All matrices have been successfully converted and saved.');
figure
imagesc(clipped_R)
colorbar;


%% Binarization fcb
%This part of the code was written to binarize the FC connectomes, it was
%never implemented in the thesis

source_dir = 'results/fc_dk/weight';
output_dir = 'results/fc_dk/binary';

matrices = cell(1, 100);

for i = 1:100
    
    %Paths
    filename = sprintf('sub-%02d_fcw_connectome.txt', i);
    txt_file_path = fullfile(source_dir, filename);

    if exist(txt_file_path, 'file')

        matrices{i} = readmatrix(txt_file_path);
        matrices{i} = threshold_proportional(matrices{i},0.7);

        % Display subject information
        disp(['Processing Subject : ', num2str(i)]);
       
    else
        disp(['File does not exist: ', txt_file_path]);
    end
    
    txt_file_path = fullfile(output_dir, sprintf('sub-%02d_fcb_connectome.txt', i));
    writematrix(matrices{i}, txt_file_path, 'Delimiter', 'tab');
end



%--------------------------------------------------------------------------------------
%Version v1.0.
%--------------------------------------------------------------------------------------
%Get the lastest version at:
%--------------------------------------------------------------------------------------
%script by Alejandro Garma Oehmichen
%--------------------------------------------------------------------------------------
