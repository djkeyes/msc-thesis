


filename = '~/dev/msc-thesis/cpp/build/points_normals_pfhs.txt';

fid = fopen(filename);
% linelen = 4 + 4 + 125;
% lineformat = [repmat('%f ', 1, linelen-1) '%f'];
lineformat = ['(%f,%f,%f - %f) (%f,%f,%f - %f) (' repmat('%f, ', 1, 124) '%f)'];
contents = textscan(fid, lineformat);
fclose(fid);


positions = [contents{1} contents{2} contents{3}];
intensity = contents{4};
normals = [contents{5} contents{6} contents{7}];
curvatures = contents{8};
features = horzcat(contents{9:end});


valid = ~isnan(contents{5});
positions = positions(valid, :);
intensity = intensity(valid, :);
normals = normals(valid, :);
curvatures = curvatures(valid, :);
features = features(valid, :);
numValid = nnz(valid);


% this is a lot of points, so subsample a few to plot

subsamplePairs = true;
if (subsamplePairs)
    % subsample onle the pairs
    numPairsToSample = 250000;
    
    numPairs = (numValid-1)*numValid/2;
    % is there a way to sample this without replacement, but in linear
    % time?
    subsampledIndices = randsample(numPairs, numPairsToSample, true);
    
    euclidianDistance = zeros(1, numPairsToSample);
    featureDistance = zeros(1, numPairsToSample);
    
    for i=1:numPairsToSample
        idx = subsampledIndices(i);
        
        first = floor((idx-1)/numValid)+1;
        second = mod(idx-1, numValid)+1;
        
        if(first <= second)
            first = numValid - first;
            second = numValid + 1 - second;
        end
        
        euclidianDistance(i) = norm(positions(first,:) - positions(second,:));
        featureDistance(i) = norm(features(first,:) - features(second,:));
    end
else
    
    % subsample points, and check all connections within the points
    
    numToSubsample = 400;
    subsampledIndices = randsample(numValid, numToSubsample);

    euclidianDistance = [];
    featureDistance = [];
    for i=1:numToSubsample
        for j=1:i-1
            first = subsampledIndices(i);
            second = subsampledIndices(j);

            euclidianDistance(end+1) = norm(positions(first,:) - positions(second,:));
            featureDistance(end+1) = norm(features(first,:) - features(second,:));
        end
    end

end

scatter(euclidianDistance, featureDistance, '.');
title('Euclidian vs PFH distances for pairs of points');
xlabel('distance between 3D points');
ylabel('distance between PFH features');