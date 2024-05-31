function processJson(inputFileName, outputFileName)
    % Open the input file
    fp = fopen(inputFileName, 'r');
    fcont = fread(fp);
    fclose(fp);

    % Convert content to character array
    cfcont = char(fcont');

    % Find indices for slice timing information
    i1 = strfind(cfcont, 'SliceTiming');
    i2 = strfind(cfcont(i1:end), '[');
    i3 = strfind(cfcont((i1+i2):end), ']');
    cslicetimes = cfcont((i1+i2+1):(i1+i2+i3-2));

    % Extract slice times as numbers
    slicetimes = textscan(cslicetimes, '%f', 'Delimiter', ',');
    [sortedslicetimes, sindx] = sort(slicetimes{1});
    mb = length(sortedslicetimes) / (sum(diff(sortedslicetimes) ~= 0) + 1);
    slspec = reshape(sindx, [mb, length(sindx)/mb])' - 1;

    % Write the output to a file
    dlmwrite(outputFileName, slspec, 'delimiter', ' ', 'precision', '%3d');
end
