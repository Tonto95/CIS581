function [H, WidthOrigin, HeightOrigin]=model_homography(image1, image2)


points1 = detectHarrisFeatures(rgb2gray(image1));
points2 = detectHarrisFeatures(rgb2gray(image2));

[features1,valid_points1] = extractFeatures(rgb2gray(image1),points1);
[features2,valid_points2] = extractFeatures(rgb2gray(image2),points2);

indexPairs = matchFeatures(features1,features2);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);



CONTROL_K = 100;
e = 0.6;
count = 0;
%currentMatches = [];
currentH = zeros(3,3);

for k=1:CONTROL_K
    sample = ceil(rand(1) * size(matchedPoints2.Location, 1));    
    deltaX = matchedPoints2.Location(sample, 1) - matchedPoints1.Location(sample, 1);
    deltaY = matchedPoints2.Location(sample, 2) - matchedPoints1.Location(sample, 2);
    H=[1 0 deltaX; 0 1 deltaY; 0 0 1];
    currentCount = 0;
    %clear matchIndexes;
    
    for i = 1:size(matchedPoints1.Location, 1)
        lhs = H * [matchedPoints1.Location(i, 1); matchedPoints1.Location(i, 2); 1];
        xnew = lhs(1);
        ynew = lhs(2);
        
        ssd = (matchedPoints2.Location(i, 1) - xnew)^2 + (matchedPoints2.Location(i, 2) - ynew)^2;
        if ssd < e
            currentCount = currentCount + 1;
            %matchIndexes(currentCount, 1) = i;
        end
    end
    
    if currentCount > count
        count = currentCount;
        %currentMatches = matchIndexes;
        currentH = H;
    end
end

WidthOrigin = -(currentH(2,3));
HeightOrigin = -(currentH(1,3));

H = currentH;
            
        
        
        
        
    
    