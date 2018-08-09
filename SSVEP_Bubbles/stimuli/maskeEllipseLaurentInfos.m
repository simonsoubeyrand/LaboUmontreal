load maskellipse.mat
% feature coordinates and sizes
eye_hl = 203; % horizontal position of left eye
eye_hr = 365; % horizontal position of right eye
eye_v = 382; % vertical position of both eyes
eye_r = 66; % circular radius of both eyes
mouth_h = 283; % horizontal position of mouth
mouth_v = 535; % vertical position of mouth
mouth_rh = 100; % horizontal elliptical radius of mouth
mouth_rv = 41; % vertical elliptical radius of mouth
face_ch = 283; % horizontal center of face  % don't change
face_cv = 435; % vertical center of face
face_rh = 144; % horizontal radius of face
face_rv = 187; % vertical radius of face
mask_smoothing = 10; % std of mask smoothing gaussian kernel
facemask_smoothing = 10; % std of facemask smoothing gaussian kernel
feat_backgr_color = meancol; % either meancol or any rgb triplet (0-1)
im_backgr_color = [0.5 0.5 0.5]; % rgb triplet
imratio = fsize/(2*face_rh); % ratio to put face width at 256 pixels (6 degs)

% image with background color
meancolim = single(repmat(shiftdim(repmat(feat_backgr_color',[1 isize isize]),1),[1 1 1 nfeatures]));
backgrcolim = uint8(zeros(isize));
for ii = 1:3, backgrcolim(:,:,ii) = im_backgr_color(ii)*255; end

% create face mask
facemask = zeros(ySize,xSize);
[xx,yy] = meshgrid(1:xSize,1:ySize);
facemask((xx-face_ch).^2./(face_rh.^2) + (yy-face_cv).^2./(face_rv.^2) < 1) = 1;
facemask = SmoothCi(facemask, facemask_smoothing);
facemask(facemask<0.01) = 0;
facemask = imresize(facemask(156:end-48,4:end),imratio); % crop so that 256x256 and center
facemask = repmat(facemask,[1 1 3]);
facemask_c = repmat(facemask,[1 1 1 3]);