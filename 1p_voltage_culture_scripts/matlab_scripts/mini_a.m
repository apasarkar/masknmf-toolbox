
%% Load imaging video
disp('Load imaging video');


% For Luminos
% cameradata = load('output_data.mat');
% nrow=double(cameradata.Device_Data{1, 2}.ROI(4));
% ncol=double(cameradata.Device_Data{1, 2}.ROI(2));
% movie_path = uigetdir();
nrow=256;
ncol=568;
frame_path = "/media/app2139/SanDisk_1/Jinfan_NGN2/frames1.bin";
movie = -double(readBinMov(frame_path, nrow, ncol)); % Load Movie, flip polarity

% Parameters
height = size(movie,1); % Height of the field of view
width = size(movie,2); % Width of the field of view
frame_count = size(movie,3); % Number of frames in the movie
fps = 1 / 1.25 * 1000; % Frames per second
time = (0:frame_count-1) / fps; % Time vector

%% Motion corretion (Optional for slice or culture)
% 
% % Motion correction using Farneback method
% ref_frame = mean(movie(:,:,1:end/10), 3); % Reference frame
% opticFlow = opticalFlowFarneback('NumPyramidLevels', 3, 'PyramidScale', 0.5, 'NumIterations', 3);
% 
% % Loop through each frame
% for t = 1:frame_count
%     flow = estimateFlow(opticFlow, movie(:,:,t));
% 
%     % Create grids for original and shifted coordinates
%     [X, Y] = meshgrid(1:width, 1:height);
%     shiftedX = X + flow.Vx;
%     shiftedY = Y + flow.Vy;
% 
%     % Warp the image using the flow vectors
%     movie(:,:,t) = interp2(X, Y, movie(:,:,t), shiftedX, shiftedY, 'linear', 0);
% end
%% Lag 1 temporal autocorrelation and regionprops
disp('Lag 1 temporal autocorrelation and regionprops');

autoCorr = sum(movie(:,:,1:end-1).*movie(:,:,2:end),3)/frame_count;
%binary_threshold = 4e6; %Threshold for generating binary map, setting by
%hand
%or
%setting based on histogram,threshold 95% brightness
his_values = histogram(autoCorr(:),1000).Values;
his_binedges = histogram(autoCorr(:),1000).BinEdges;
for z = 1:size(his_values,2)
    ratio = sum(his_values(:,1:z),2)/sum(his_values,2);
    if ratio > 0.96
        break
    end
end
binary_threshold = his_binedges(z-1);

binary_mask = zeros(size(autoCorr));
binary_mask(autoCorr > binary_threshold) = 1;


labeled_mask = logical(binary_mask);
regions = regionprops(labeled_mask, 'Area', 'BoundingBox', 'Centroid', 'PixelIdxList', 'PixelList');
spatialFootprints = cell(length(regions), 1);
for a = 1:length(regions)
    spatialFootprints{a} = regions(a).PixelList;    
end

close

% Display the binary mask with labeling
figure;
imshow(binary_mask);
hold on;
% Plot the centroids and surroundings of the regions
for b = 1:length(regions)
    % Plot centroid
    plot(regions(b).Centroid(1), regions(b).Centroid(2), 'r*');
    bbox = regions(b).BoundingBox;
    rectangle('Position', [bbox(1), bbox(2), bbox(3), bbox(4)], 'EdgeColor', 'g', 'LineWidth', 1);
    text(regions(b).Centroid(1)+5, regions(b).Centroid(2)+5,num2str(b),'Color', [1 0 0])
end
hold off;
title('Detected Regions with Centroids and Bounding Boxes');
savefig('Detected Regions with Centroids and Bounding Boxes');


%% Apply moving median filter
disp('Apply moving median filter')

filter_window = 8;% 10ms window
filtered_movie = movie - movmedian(movie, filter_window, 3); 
%Truncate begining and ending frames to avoid filter-artifacts
filtered_movie(:,:,[1:round(filter_window/2), round(end-filter_window/2+1):end]) = zeros(height,width,filter_window);
filtered_movie_max = max(filtered_movie,[],3);

%% Apply Difference Of Guassion Filter
disp('Apply Difference Of Guassion Filter')

sigma1 = 1; % Parameters for DOG filter, 
sigma2 = 3;
filter_width = 19;

% Generate a DOG filter with 0 averaged area
area_ratio = trapz(1:filter_width,gaussmf(1:filter_width,[sigma1 (1+filter_width)/2])/trapz(1:filter_width,gaussmf(1:filter_width,[sigma2 (1+filter_width)/2])));
dog_filter = gaussmf(1:filter_width,[sigma1 (1+filter_width)/2]) - gaussmf(1:filter_width,[sigma2 (1+filter_width)/2]) * area_ratio;
dog_filter = dog_filter/max(dog_filter);
dog_filter = reshape(dog_filter,1,1,[]);
dog_filtered_movie = imfilter(filtered_movie, dog_filter);
dog_filtered_movie_max = max(dog_filtered_movie,[],3);


%% Mini ALI for each Region
disp('Mini ALI for each Region')

% Threshold factor for AP detection
threshold_factor = 5; 
% Parameters for DBSCAN clustering
epsilon = 0.75; % Maximum distance between points to be considered in the same cluster
minPts = 2; % Minimum number of points to form a cluster
patch_size = [27, 27]; % Size of patch for each region
APs = [];
COMs = [];
footprint_selection = [];
footprint = [];
footprint_center = [];

for c = 1: size(regions,1)
    region_AP = [];% Record the APs in current region
    % Report the progress
    disp(['analyzing the no.' num2str(c) ' region of ' num2str(size(regions,1)) ' regions']);
    
    %==================================================
    % Decide the patch size for each footprint. Either a 27X27 patch or a bouding box
    min_X = min(floor(regions(c).Centroid(1,2)-patch_size(2)*0.5),floor(regions(c).BoundingBox(1,2)+0.5));
    min_Y = min(floor(regions(c).Centroid(1,1)-patch_size(1)*0.5),floor(regions(c).BoundingBox(1,1)+0.5));
    max_X = max(floor(regions(c).Centroid(1,2)+patch_size(2)*0.5),floor(regions(c).BoundingBox(1,2)+0.5+regions(c).BoundingBox(1,4)));
    max_Y = max(floor(regions(c).Centroid(1,1)+patch_size(1)*0.5),floor(regions(c).BoundingBox(1,1)+0.5+regions(c).BoundingBox(1,3)));
    patch_X = max(min_X,1):min(max_X,height);
    patch_Y = max(min_Y,1):min(max_Y,width);
    % Record the up left coordinate of each patch
    patch_coordinate = [patch_X(1) patch_Y(1)];

    %==================================================
    % Select the binary mask for analyzation
    cell_mask = spatialFootprints{c};
    cell_mask = cell_mask(:,[2 1]);
    selection_map = zeros(height,width);
    for d = 1:size(cell_mask,1)
        selection_map(cell_mask(d,1),cell_mask(d,2)) = 1;
    end
    selection_map_patch = selection_map(patch_X,patch_Y);

    %==================================================
    % Use DOG filtered movie for AP detection
    patch_movie = dog_filtered_movie(patch_X,patch_Y,:);
    for e = 1:frame_count
        patch_movie(:,:,e) = patch_movie(:,:,e).* selection_map_patch;
    end
    patch_movie (patch_movie ==0) = NaN;
    patch_trace = tovec(mean(patch_movie,[1 2],"omitnan"));
    % Set Threshold
    std_noise = std(patch_trace(10:350),"omitnan");% standard deviation of baseline without blue illumination
    threshold = threshold_factor .* std_noise;
    patch_trace(patch_trace<threshold)=0;
    % Peak detection in time and space
    [peaks, peak_time] = findpeaks (patch_trace);
    for f = 1:size(peak_time,2)
        peak_value = max(patch_movie(:,:,peak_time(1,f)),[],'all');
        [peak_position_x,peak_position_y] = find (patch_movie(:,:,peak_time(1,f))== peak_value,1); 
        region_AP = [region_AP; peak_position_x+patch_coordinate(1)-1,peak_position_y+patch_coordinate(2)-1,peak_time(1,f),peaks(1,f)];
    end      
    % Remove the timepoints that are too close (within 10 frames)
    if ~isempty(region_AP)
        minimal_interval = 10;
        peak_time = intersect(uniquetol(region_AP(:,3), minimal_interval,'highest', 'DataScale', 1),uniquetol(region_AP(:,3), minimal_interval,'lowest', 'DataScale', 1)); 
        [~,AP_exclude] = setdiff(region_AP(:,3),peak_time);
        region_AP(AP_exclude,:) = [];   
    end

    % Save all APs
    if ~isempty(region_AP)
        APs = [APs; region_AP];
    end

    %==================================================
    % If no AP was detected, skip the loop and use region map as footprint
    if isempty(region_AP)
        map_noAP = zeros(height,width);
        for g = 1:size(cell_mask,1)
            map_noAP(cell_mask(g,1),cell_mask(g,2))=1;
        end
        footprint_region = filtered_movie_max.*map_noAP;
        footprint_weight = diag(footprint_region(cell_mask(:,1),cell_mask(:,2)));
        footprint_x = sum(cell_mask(:,1).*footprint_weight.*footprint_weight)/sum(footprint_weight.*footprint_weight);
        footprint_y = sum(cell_mask(:,2).*footprint_weight.*footprint_weight)/sum(footprint_weight.*footprint_weight);
        % Save footprint and footprint center
        footprint = cat(3,footprint,footprint_region);
        footprint_center = [footprint_center;footprint_x,footprint_y];
        continue
    end

    %==================================================
    % Calculate Center Of Mass for each region
    region_COMs = zeros (size(region_AP,1),3);
    region_COMs(:,3) = region_AP(:,3);
    stack_APs = zeros(size(patch_X,2),size(patch_Y,2),size(region_AP,1));
    for h = 1:size(region_AP,1)
        stack_APs(:,:,h) = max(filtered_movie(patch_X,patch_Y,region_AP(h,3)-3:region_AP(h,3)+3),[],3);
    end
    % Using SVD to remove noise
    stack_APs = reshape(stack_APs,[],size(region_AP,1));
    [U,S,V] = svds(stack_APs,25);
    stack_APs = U*S*V';
    stack_APs = reshape(stack_APs,size(patch_X,2),size(patch_Y,2),[]);
    for h = 1:size(region_AP,1)
        % Generate a small surrounding patch for calcuating COMs
        small_patch_x = tovec(max(region_AP(h,1)-patch_coordinate(1)+1-3,1):min(region_AP(h,1)-patch_coordinate(1)+1+3,size(patch_X,2)));
        small_patch_y = tovec(max(region_AP(h,2)-patch_coordinate(2)+1-3,1):min(region_AP(h,2)-patch_coordinate(2)+1+3,size(patch_Y,2)));
        % Pick out the frames of each peak
        small_patch = stack_APs (small_patch_x,small_patch_y,h);
        
        %Calculate the weighted centers of the brightest 10 pixels
        pixel_pos = zeros (15,2);
        weight = zeros (15,1);
        for i = 1:15
            wt = max(small_patch (:));
            [pixel_pos(i,1), pixel_pos(i,2)]= find (small_patch == wt,1);
            small_patch (pixel_pos(i,1), pixel_pos(i,2)) = NaN;
            weight (i,:) = wt*(wt>0);
        end
        pixel_pos_weighted = [sum(weight.*weight.*pixel_pos(:,1))/sum(weight.*weight);sum(weight.*weight.*pixel_pos(:,2))/sum(weight.*weight)];
        region_COMs(h,1) = pixel_pos_weighted (1,:)+small_patch_x(1,1)+patch_coordinate(1,1)-2; region_COMs(h,2) = pixel_pos_weighted (2,:)+small_patch_y(1,1)+patch_coordinate(1,2)-2;
    end
    region_COMs (isnan(region_COMs(:,3)),:) = [];

    % If no COMs was detected, skip the loop and use region map as footprint
    if isempty(region_COMs)
        map_noCOM = zeros(height,width);
        for g = 1:size(cell_mask,1)
            map_noCOM(cell_mask(g,1),cell_mask(g,2))=1;
        end
        footprint_region = filtered_movie_max.*map_noAP;
        % Save footprint and footprint center
        footprint = cat(3,footprint,footprint_region);
        footprint_center = [footprint_center;regions(c).Centroid(1,2),regions(c).Centroid(1,1)];
        continue
    end


    % Save all COMs
    if ~isempty(region_COMs)
        COMs = [COMs; region_COMs];
    end

    

    %==================================================
    % Spike Clustering for each region with DBSCAN
    % Perform DBSCAN clustering
    clustering = dbscan(region_COMs(:, 1:2), epsilon, minPts);
    % Skip the loop if there is no clustering and use region as footprint
    if clustering==-1
        map_noclustering = zeros(height,width);
        for j = 1:size(cell_mask,1)
            map_noclustering(cell_mask(j,1),cell_mask(j,2))=1;
        end
        footprint_region = filtered_movie_max.*map_noclustering;
        % Save footprint and footprint center
        footprint = cat(3,footprint,footprint_region);
        footprint_center = [footprint_center;regions(c).Centroid(1,2),regions(c).Centroid(1,1)];
        continue
    end
    % Store the results
    cluster_time = {};
    for k = 1:max(clustering)
    cluster = region_COMs(clustering(:,1) == k,3);
    cluster_time = [cluster_time;cluster]; 
    end
    % Record the number of clusters
    num_clusters = size (cluster_time,1);

    %==================================================
    % Extraction of footpritnts
    footprint_region = zeros(height,width,num_clusters);
    cluster_center = zeros(size (cluster_time,1), 2);
    for l = 1:size (cluster_time,1)
        extracted_stacks = dog_filtered_movie(:,:,cluster_time{l,1});
         extracted_footprint_raw = mean (extracted_stacks,3);
        [peakx,peaky]= find((extracted_footprint_raw.*selection_map)==max(extracted_footprint_raw.*selection_map,[],'all'),1);
        % Generate a footprint in neighbouring 11x11 pixels of peaks
        selection = zeros (height,width);
        selection_x = max(peakx-5,1):min(peakx+5,height);
        selection_y = max(peaky-5,1):min(peaky+5,width);
        selection (selection_x,selection_y)=1;
        footprint_region(:,:,l) = extracted_footprint_raw.*selection;
        %save the postiton of peaks for each footprint
        cluster_center(l,:) = [peakx peaky];
    end
    footprint_center = [footprint_center;cluster_center];
   
    % Save all footprints
    if ~isempty(selection)
        footprint_selection = cat(3,footprint_selection,selection);
        footprint = cat(3,footprint,footprint_region);
    end

end

%% Find pseudoinverse and left multiply to get traces for each footprint
disp('Find pseudoinverse and left multiply')

% Flatten the corrected movie to prepare for matrix operations
flattened_movie = reshape(movie, [], size(movie,3));
%flattened_movie_filtered = reshape(filtered_movie, [], size(movie,3));
flattened_footprint = reshape(footprint, [], size(footprint,3));


% Compute the pseudoinverse of the spatial footprints
footprint_pseudoinverse = pinv(flattened_footprint);

% Decompose the movie into cell traces using the backslash operator
cell_traces = footprint_pseudoinverse * flattened_movie;

% Reshape cell traces to have dimensions (num_cells, frame_count)
cell_traces = reshape(cell_traces, size(footprint,3), size(movie,3));

% Normalize each cell trace by its maximum value
normalized_traces = bsxfun(@rdivide, cell_traces, max(cell_traces, [], 2));
offset_traces = bsxfun(@plus, normalized_traces, (1:size(normalized_traces,1))' - 0.5);

%% Save data
%save ("filtered_movie.mat","filtered_movie");
%save ("dog_filtered_movie.mat","dog_filtered_movie");
save ("ALI_Result.mat","footprint","footprint_center","cell_traces");
%save ("cell_traces.mat","cell_traces");
%% Generate scatterplot figure for APs
% max_frame = dog_filtered_movie_max;
% 
% figure_handle = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
% imshow(max_frame, [], 'InitialMagnification', 'fit');
% hold on;
% scatter(APs(:, 2), APs(:, 1), 15, 'r', 'filled', 'MarkerEdgeColor', 'k');
% title('APs');
% hold off;   


%% Generate scatterplot figure for COMs
max_frame = dog_filtered_movie_max;

figure_handle = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
imshow(max_frame, [], 'InitialMagnification', 'fit');
hold on;
scatter(COMs(:, 2), COMs(:, 1), 15, 'r', 'filled', 'MarkerEdgeColor', 'k');
title('COMs');
hold off;   
savefig('COMs');
%% Plot traces
cmap = jet(size(normalized_traces,1));
time = [1:size(normalized_traces,2)];
offset_traces = normalized_traces;
figure;
hold on;
% Plot each trace with a different color
for i = 1:size(normalized_traces,1)
    plot(time, rescale(offset_traces(i, :))+i-0.5, 'Color', cmap(i, :), 'LineWidth', 1.5);
end
% Set axis labels
xlabel('Time (s)');
ylabel(' Cell Traces');
% Set y-axis ticks to indicate trace numbers
yticks(0.5:5:size(normalized_traces,1));
yticklabels(arrayfun(@num2str, 1:5:size(normalized_traces,1), 'UniformOutput', false));
xlim([0; max(time)]);
ylim([0; size(normalized_traces,1)+1]);
% Set the title
title('Normalized Cell Traces from Original Video');
% Hold off to stop adding to the current plot
hold off;
savefig('Normalized Cell Traces from Original Video');
%% Plot center of regions
max_frame = dog_filtered_movie_max;
cmap = jet(size(footprint,3));

figure;
hold on;
axis equal;
imagesc(max_frame);
hold on;
% Plot circles at the COM positions
for t = 1:size(footprint_center,1)
    % Draw a circle of radius 10
    plot_pos = footprint_center(t, :);
    viscircles([plot_pos(2), plot_pos(1)] , 3, 'Color', cmap(t, :), 'LineWidth', 1.5);
    text(plot_pos(2)+5, plot_pos(1)+5,num2str(t),'Color', cmap(t, :))
end
axis('tight');
set(gca, 'YDir','reverse')
title('Center of cell regions');
% Hold off to stop adding to the current plot
hold off;
savefig('Center of cell regions');