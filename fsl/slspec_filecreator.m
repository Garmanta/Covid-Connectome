%This objective of this code is to generate a proper slspec.txt file given
%a .json file. 
%Code obtained from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#How_should_my_--slspec_file_look.3F
%The output consists of the order under which the MRI scanned the
%slices

fp = fopen('sub-05_task-rest_bold.json','r');
fcont = fread(fp);
fclose(fp);
cfcont = char(fcont');
i1 = strfind(cfcont,'SliceTiming');
i2 = strfind(cfcont(i1:end),'[');
i3 = strfind(cfcont((i1+i2):end),']');
cslicetimes = cfcont((i1+i2+1):(i1+i2+i3-2));
slicetimes = textscan(cslicetimes,'%f','Delimiter',',');
[sortedslicetimes,sindx] = sort(slicetimes{1});
mb = length(sortedslicetimes)/(sum(diff(sortedslicetimes)~=0)+1);
slspec = reshape(sindx,[mb length(sindx)/mb])'-1;
dlmwrite('slspec.txt',slspec,'delimiter',' ','precision',c'%3d');