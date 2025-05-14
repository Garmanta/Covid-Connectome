%nbs_result_2_txt.m v1.0
%The objective of this code is to copy the NBS statistical significant
%network into a .txt file to be imported into another programs. In this
%case, the Python code named: nbsfiltering.py

%Therefore, the use case is:
%NBS --> nbs_results_2_txt.m --> nbsfiltering.py

%For this code to run correctly, one has to:
% 1.- Run NBS in the console and change the path to the correct location
% 2.- Select the appropiate connecotomes, design and contrast files.
% 3.- Obtain a valid, statistically significant network

%The input files are: NBS required files
%The output files is: nbs_connectome.txt

%Path
output_file_path = '/home/sphyrna/storage/subjects/results/nbs/sc_dk/weight/scw_nbs_connectome2.txt';

global nbs;
nbs_simplet = nbs.NBS;

adj=nbs.NBS.con_mat{1}+nbs.NBS.con_mat{1}';

%Grabs the statistically significant network.
matrix_adj = full(adj);
sum(sum(matrix_adj))

%Writes the output file
writematrix(matrix_adj, output_file_path, 'Delimiter', 'tab');



%--------------------------------------------------------------------------------------
%Version v1.0.
%--------------------------------------------------------------------------------------
%Get the lastest version at:
%--------------------------------------------------------------------------------------
%script by Alejandro Garma Oehmichen
%--------------------------------------------------------------------------------------
