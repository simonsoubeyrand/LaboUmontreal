function SSVEP_Features_Gender_v2(name, tags_hz, EEG, block)
%  Inducing steady-state visual evoked potential (SSVEP) for each principal facial feature (left eye,
%  right eye & mouth) while in a face gender discrimination task.

%   write thorough description here
%   Author:     Simon Faghel-Soubeyrand
%   Date:       feb/2018
%   Version:    1
%   Tested by:  _____________
%
%  Experimenter will need to add  :
%  1- a file_name ('string') on the first argument
%  2- a vector of 3 frequency tagg (either integer or float) : e.g. [5.8 10 6.66]
%  3- With trigger (EEG==1) or without (EEG==0)
%  4- Block number
%
%
%  SSVEP_features('Bonobo',[5.8 10 6.66],1,1) : Bonobo begin task, with
%  freq tags of 5.8, 10 and 6.66, with EEG, block 1.
%
%  Frequency of taggs will are randomnized on every trial (~70
%  sec each). Intermodulate frequencies (e.g. 5.8-6.6 Hz = 1.2) will need to
%  be away from alpha (8-12hz) AND from other taggs.

KbName('UnifyKeyNames');                            % Else right and left arrow wont work.

% put to 0 preferably !!!!!!!!!
Screen('Preference', 'SkipSyncTests', 1)            % put to 0 preferably !!!!!!!!!
% put to 0 preferably !!!!!!!!!

% Manages input(s)
if ~isstring(name)
    error('First argument must be a string (the name of the file).')
end

if size(tags_hz)~=[1,3]                                                                                                                                              -
    error('Second argument must be a vector containing 3 different frequencies for tagging.')
end

clear data
file_name=sprintf('%s_SSVEPGender_%d',name,block);


% Checks if file_name already exits
if fopen([file_name,'.mat'])>0
    reenter = input('WARNING: This filename already exists. Overwrite (y/n)? ', 's');
    if strcmp(reenter, 'n'),
        file_name = input('Enter new filename: ', 's');
    end
end

if EEG==1
    disp('trig begining')
    %Clear TTL Port and open it
    config_io;
    adress = hex2dec('378'); % Adress of port
    outp(adress, 0);
    
    % All trigger values
    TriggStim=1;
    TriggENDStim=2;
    TriggNewFace=100;
    TriggMaleFace=151;
    TriggFemaleFace=150;
    TriggRespMan=251;
    TriggRespWoman=250;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FACE STIMULI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% These path will need to be changed to that of the actual folders of the
% XP computer.


% Change according to PC
load /Users/SimonFaghel-Soubeyrand/Documents/SSVEP_Bubbles/stimuli/cropped_ims.mat
load /Users/SimonFaghel-Soubeyrand/Documents/SSVEP_Bubbles/stimuli/meanbackgr.mat
% % % % % % % % % % % % % % % % % % % %% % % % %

im = cropped_ims; % need to check images so that width of face is 256 pixels (6 deg at 76 cm)



meanIm=zeros(size(im{1}));
for ii=1:max(size(im))
    Mim{ii}=mean(im{ii},3);
    for jj=1:3,im{ii}(:,:,jj)=Mim{ii};end
    if mod(ii,2)==0
        meanIm=meanIm+double(im{ii})/255;
    end
end
meanIm=meanIm/(max(size(im))/2);
load images.mat
% load imScrambleColour.mat
load imScrambleGray.mat



isize=497;
expe.nfeatures = 3;
expe.xSize = isize;%562;
expe.ySize = isize;%762;
expe.contrast = 1;
meanbackgr=double(meanIm);%double(imresize(meanAndroALL,[isize isize]));



% feature coordinates and sizes
nfeatures = expe.nfeatures;
xSize = expe.xSize;
ySize = expe.ySize;
eye_hl = 182; % horizontal position of left eye
eye_hr = 316; % horizontal position of right eye


eye_v = 202; % vertical position of both eyes
eye_r = 66; % circular radius of both eyes


mouth_h = 247; % horizontal position of mouth
% mouth_v = 380; % vertical position of mouth
mouth_v = 339; % vertical position of mouth




mouth_rh = 100; % horizontal elliptical radius of mouth
mouth_rv = 41; % vertical elliptical radius of mouth
face_ch = 250; % horizontal center of face  % don't change
face_cv = 250; % vertical center of face
face_rh = 144; % horizontal radius of face
face_rv = 187; % vertical radius of face
mask_smoothing = 10; % std of mask smoothing gaussian kernel
facemask_smoothing = 10; % std of facemask smoothing gaussian kernel
% meancol=[0.94 0.78 0.6822]-.07; %was this
% meancol=[0.7178  0.7019 0.6868];
feat_backgr_color =[mean(meancol) mean(meancol) mean(meancol)];% meancol; % either meancol or any rgb triplet (0-1)
im_backgr_color = [0.5 0.5 0.5];




% imratio = 256/(2*face_rh); % ratio to put face width at 256 pixels (6 degs)


% image with background color
meancolim = uint8(zeros(ySize, xSize));
for ii = 1:3
    meancolim(:,:,ii) = feat_backgr_color(ii)*255;
end
backgrcolim = uint8(zeros(ySize, xSize));
for ii = 1:3
    backgrcolim(:,:,ii) = im_backgr_color(ii)*255;
end


% create feature mask
mask = zeros(nfeatures,ySize,xSize);
[xx,yy] = meshgrid(1:xSize,1:ySize);
mask(1, (xx-eye_hl).^2 + (yy-eye_v).^2 < eye_r.^2) = 1;
mask(2, (xx-eye_hr).^2 + (yy-eye_v).^2 < eye_r.^2) = 1;
mask(3, (xx-mouth_h).^2./(mouth_rh.^2) + (yy-mouth_v).^2./(mouth_rv.^2) < 1) = 1;
mask = shiftdim(mask,1);
for ii=1:nfeatures, mask(:,:,ii) = SmoothCi(squeeze(mask(:,:,ii)),mask_smoothing); end
mask(mask<0.06) = 0;


% mask for the features
ThreeFeature_mask = mask(:,:,1)+mask(:,:,2) + mask(:,:,3);
ThreeFeature_mask = repmat(ThreeFeature_mask,[1 1 3]);


% create face mask
facemask = zeros(ySize,xSize);
[xx,yy] = meshgrid(1:xSize,1:ySize);
facemask((xx-face_ch).^2./(face_rh.^2) + (yy-face_cv).^2./(face_rv.^2) < 1) = 1;
facemask = SmoothCi(facemask, facemask_smoothing);
facemask = repmat(facemask,[1 1 3]);
facemask(facemask<0.01) = 0;


% background for all stimuli
backgr_face =(repmat((-sum(mask,3)+1),[1 1 3])) .* meanbackgr;
backgr_face = double(backgr_face) .* facemask;
%backgr_feat = (repmat(sum(mask,3),[1 1 3]) .* double(meancolim)/255) .* facemask;
backgr_im = (-double(facemask)+1) .* double(backgrcolim)/255;
%backgr = backgr_feat+backgr_face+backgr_im;

rng('shuffle')                                  % uses clock to seed pseudo-random generators

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Psychtoolbox inits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AssertOpenGL;
screens=Screen('Screens');
screenNumber=max(screens);
background_color = 128;
[windowPtr,windowRect]=Screen('OpenWindow',screenNumber, background_color);
[monitorFlipInterval, nrValidSamples, stddev] = Screen('GetFlipInterval', windowPtr);
Screen('Blendfunction', windowPtr, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
Screen('TextFont',windowPtr, 'Times');
Screen('TextSize',windowPtr, 30);
Screen('TextStyle',windowPtr, 1);


white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
gray = (white+black)/2;


Priority(MaxPriority(windowPtr));
%   Currently OS/X doesn't support this function at all, and on MS-Windows
%   and GNU/Linux, only recent ATI GPU's with recent drivers do support it.
%
%   All subfunctions return an optional 'rc' return code of zero on success,
%   non-zero on error or if the feature is unsupported.
%
%   ...
%
%   Select the performance state of the GPU. 'gpuPerformance' can be set to
%   0 if the GPU shall automatically adjust its performance and power-
%   consumption, or to one of 10 fixed levels between 1 and 10, where 1 means
%   the lowest performance - and power consumption, whereas 10 means the
%   highest performance - and maximum power consumption.
%
%   If in doubt, choose 10 for best accuracy of visual stimulus onset timing,
%   0 for non-critical activities to leave the decision up to the graphics
%   driver and GPU.
rc = PsychGPUControl('SetGPUPerformance', 10);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instructions : not for now
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% XpInstruction('TestInstru_GenderSSVEP.m', windowPtr)                 % Get .m text instruction and display on different screen before begining task


%Initialize QEST parameters
Questy.tGuessSd=.04; %
Questy.pThreshold=.82; % desired probability of accuracy for threshold
Questy.beta=2; % Steepness of the sigmoid curve is typically 3.5 but differs for random sampling methods
Questy.delta=.01;% probability that the observers answers were "blind" over trials. Typically .01
Questy.gamma=.5; % Chance.
Questy.range=.15; % any guess for what this should be? for now, its from tGuess-(range/2) to tGuess+(range/2)
Questy.grain=.01;
%     response=QuestSimulate(q,intensity,tActual [,plotIt])

if block ==1 % sets QUEST's "best guess"
    
    Signal=1; % in % (% Jess: Maybe we can change this to a better estimate after pilots)
    Questy.tGuess=Signal;
    
    % Starts Quest
    Questy.q=QuestCreate(Questy.tGuess,Questy.tGuessSd,Questy.pThreshold,Questy.beta,Questy.delta,Questy.gamma,[],Questy.range);
    
else % sets QUEST's "best guess" threshold value as the number of features from last run
    
    LastPath = fullfile(resultsDir,sprintf('parameters_%s_run%d.mat',sID,block-1));
    lastRun=load(LastPath);
    lastIntensities=find(last.q.intensity(find(abs(last.q.intensity(:))>0)));
    lastIntensitySignal=lastIntensities(end);
    Questy.tGuess=lastIntensitySignal;
    
    if params.flushQuest
        % Starts Quest
        Questy.q=QuestCreate(Questy.tGuess,Questy.tGuessSd,Questy.pThreshold,Questy.beta,Questy.delta,Questy.gamma,[],Questy.range);
    else
        % continues Quest from last run, including trial history.
        Questy.q= lastRun.outparams.QUEST;
    end
    clear lastRun
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Key constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
key_man = 'x';                             % target present in main task / horizontal in neutral task
key_woman = 'z';                             % target present in main task / horizontal in neutral task
key_quit = 'ESCAPE';                                    % to quit the task


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Timing constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
target_blank_interval = 0.5;                                                        % in seconds
blank_stimulus_interval = 0.3;                                                      % in seconds
blank_target_interval = 0.75;                                                       % in seconds
jitter_coeff = [0 0 0.375];                                                         % max jitter in s for presentation of target, blank before stimulus and blank after stimulus



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fixation constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xx_center = windowRect(3)/2;
yy_center = windowRect(4)/2;
fixation_width = 2;                                                                 % in pixels
fixation_length = 20;                                                               % in pixels
fixation_cross_base = [[-fixation_length/2 fixation_length/2-1 0 0];
    [0 0 -fixation_length/2 fixation_length/2-1];];                                 % fixation cross lines coordinates centered on 0,0 (upper, leftmost)
fixation_center = [[xx_center xx_center xx_center xx_center];
    [yy_center yy_center yy_center yy_center];];                                    % center of screen arranged in matrix form for later use
fixation_cross_equal = 1.25*fixation_cross_base + fixation_center;                  % 1.25 times larger fixation cross lines of same length at the center of screen;
% to mask any motion that would result from the change of only one line length
fixation_color_neutral = [0 0 255 128];                                             % color quadruplet to use during the target presentation time in neutral trials
fixation_color = [255 255 0 80];                                                   % color quadruplet to use at all other time


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Object & XP constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nObjects = 264; %%%%% % nb of different stimuli
nIDs = 66;
nExpr = 2;
nGenders = 2;
nTypes = nExpr * nGenders;
nTrialsBlock = 2; % nb of items in a block %25
nBlocks = 20; % nb of blocks %20
ntrials = nTrialsBlock * nBlocks; %nb of items total


% viewing_distance_cm = radius_cm/atan(radius_deg*pi/180);    % in cm; must adjust viewing distance accordingly
nb_objects = 3;                                             % number of objects


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defines experiment parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nb_trialsTot = 4;

length_trial=70;
ISImin=.3;
ISImax=2;
ntrialsGender=ceil(length_trial/mean([ISImin ISImax]))+20;
jitter_coeff=.03;

for trial = 1:nb_trialsTot
    
    % Stimulus parameters
    
    data{trial}.max_rt = 2;%length_trial;                                                                              % maximum response time
    data{trial}.jitter = jitter_coeff .* rand(1,3);                                                      % jitter in s for presentation of target, blank before stimulus and blank after stimulus
    data{trial}.accuracy = nan;
    data{trial}.max_ISI=1.25;
    
    for ii=1:ntrialsGender
        data{trial}.Gender(ii)    = 0.5<round(rand(1));
        data{trial}.Flipping(ii)  = 0.5<round(rand(1));
    end
    
    
    TagIndex=randperm(nb_objects);% this randomizes the freq tag assigment on every trials
    
    for ii = 1:nb_objects
        data{trial}.freq{ii} = tags_hz(TagIndex(ii));       % Tag for left right eye, and mouth
    end
    
    
end

Screen('DrawText', windowPtr, 'Man (x) or Woman (z) ?', xx_center-190, yy_center, [0,0,0,255]);
Screen('Flip', windowPtr);
KbWait;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HideCursor;

% stimuli info matrix
stim = zeros(nObjects,5);
stim(:,1) = 1:nObjects; % object
aa = repmat([1:nIDs], [2 1]);
stim(:,2) = repmat(aa(:), [2 1]); % ID
stim(:,3) = [2*ones(nIDs,1); ones(nIDs,1); 2*ones(nIDs,1); ones(nIDs,1)]; % gender 1=H 2=F
stim(:,4) = repmat([2; 1], [nIDs*2 1]); % expr 1=N 2=H
stim(:,5) = (2*stim(:,3) + stim(:,4))-2; % types 1=HN 2=HH 3=FN 4=FH

% list of stimuli for trials (picked randomly) : only happy
ind_happy=find(mod(stim(:,1),2)==1);

% list of stimuli for trials (picked randomly) : male or female
ind_male=find((stim(:,3))==1&mod(stim(:,1),2)==0);
ind_female=find((stim(:,3))==2&mod(stim(:,1),2)==0);

randInd=randperm(length(ind_male));
xpstim_female=ind_female(randInd);
xpstim_male=ind_male(randInd);


border=-10;


tags_rect_im = cell(size(tags_hz));
tags_rect_sc = cell(size(tags_hz));
tags_rect_im{1} = [(eye_hl-eye_r)-border (eye_v-eye_r)-border (eye_hl+eye_r)+border (eye_v+eye_r)+border ];                                              % in image coordinate system
tags_rect_im{2} = [(eye_hr-eye_r)-border (eye_v-eye_r)-border (eye_hr+eye_r)+border (eye_v+eye_r)+border ];


tags_rect_im{3} =[(mouth_h-mouth_rh)-border (mouth_v-mouth_rv)-border (mouth_h+mouth_rh)+border (mouth_v+mouth_rv)+border ];


for ii = 1:length(tags_hz),
    tags_rect_sc{ii} = tags_rect_im{ii} + ([xx_center-xSize/2 yy_center-ySize/2 xx_center-xSize/2 yy_center-ySize/2]);         % in screen coordinate system
end

trigRespCounter=1;
for trial = 1:nb_trialsTot
    
    
    GenderTrials=1; % this here is this counter for the number of face gender categorized (which depends on RT for each participants
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Quick pause each trial
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(trial,1)==0 && trial~=1
        
        acc_lastblock=data{trial-1}.accuracy(:);
        RT_lastblock=data{trial-1}.response_time(:);
        
        
        Screen('DrawText', windowPtr, 'Petite pause! Relaxez, puis appuyez espace pour continuer', xx_center-300, yy_center, [0,0,0,255]);
        Screen('DrawText', windowPtr, sprintf('RT moyen : %1.3f', median(RT_lastblock)), xx_center-300, yy_center+100, [0,0,0,255]);
        Screen('DrawText', windowPtr, sprintf('accuracy : %.3f, noise:%.2f', mean(acc_lastblock),1-Signal), xx_center+100, yy_center+100, [0,0,0,255]);
        Screen('Flip', windowPtr);
        KbWait;
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Draw larger fixation cross with lines of equal length
    Screen('DrawLines', windowPtr, fixation_cross_equal, fixation_width, fixation_color, [], 1);
    Screen('Flip', windowPtr);
    
    
    WaitSecs(blank_stimulus_interval+data{trial}.jitter(2));
    
    
    % Draws objects and makes features oscillate until response
    startSecs = GetSecs;
    vbl = startSecs;
    
    if EEG==1
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %  TRIGGER for beginning of stimulation (== 1)
        outp(adress,TriggStim);
        WaitSecs(0.01);
        outp(adress,0);
        % % % % %
        
    end
    
    BckImage= double(imresize(meanAndroALL,[isize isize]));%(double(im{xpstim(trial,1)})/255);
    
    % % % % % % %could maybe throw this out % % % % % % % % % % % % % %
    im_feat = zeros(expe.ySize, expe.xSize, 3, expe.nfeatures);
    backgr_feat = zeros(expe.ySize, expe.xSize, 3, expe.nfeatures);
    for feat = 1:expe.nfeatures
        im_feat(:,:,:,feat) = repmat(mask(:,:,feat),[1 1 3]) .*  BckImage .* facemask;
        backgr_feat(:,:,:,feat) = repmat(mask(:,:,feat),[1 1 3]) .* (double(meancolim)./255) .* facemask;
        back(:,:,feat)=ThreeFeature_mask(:,:,feat).*feat_backgr_color(feat);
    end
    backgroudIm_clean= (backgr_face+backgr_im+back.*facemask)*255;
    
    for ii=1:3,backgroudIm(:,:,ii)=(backgroudIm_clean(:,:,ii));end
    
    %     backgroudIm=noisy_bit(backgroudIm, 256);
    
    
    if data{trial}.Gender(GenderTrials) == 1
        whichFace=xpstim_male(GenderTrials,1);
    else
        whichFace=xpstim_female(GenderTrials,1);
    end
    data{trial}.whichFace(GenderTrials)=whichFace;
    
    for ii=1:3
        Stimulus(:,:,ii)=(double(im{whichFace}(:,:,ii))/255);
    end
    
    
    
    % To flip or not to flip?
    if data{trial}.Flipping(GenderTrials)
        for ii=1:3,Stimulus(:,:,ii)=fliplr(Stimulus(:,:,ii));end
    end
    
    if EEG==1 && trigRespCounter~=1 && GetSecs-RespSecs>data{trial}.max_ISI
        if data{trial}.Gender(GenderTrials) == 1 % which Gender is it (1==Man,0==woman)
            TriggWhichGender=151;
        else
            TriggWhichGender=150;
        end
        
        % %  TRIGGER for Gender (151==man, 150==woman)
        outp(adress,TriggWhichGender)
        WaitSecs(0.004)
        outp(adress,0)
    end
    
    
    % Starts everything to new for the SSVEP trial
    trigRespCounter=1;
    StartSecs=GetSecs;
    countSecs=StartSecs; % Time from stimulus counter
    RespSecs=countSecs; % Time from response pressed counter
    while GetSecs-startSecs <=length_trial
        
        
        if trigRespCounter~=1 && GetSecs-RespSecs>data{trial}.max_ISI
            
            GenderTrials=GenderTrials+1;
            trigRespCounter=1;
            countSecs=GetSecs;
            Signal=QuestMean(Questy.q);
            
            
            if GenderTrials>1
                data{trial}.accuracy(GenderTrials-1)
                
            end
            
            
            clear backgroudIm
            for ii=1:3,backgroudIm(:,:,ii)=(backgroudIm_clean(:,:,ii));end
            
            
            if data{trial}.Gender(GenderTrials) == 1
                whichFace=xpstim_male(GenderTrials,1);
            else
                whichFace=xpstim_female(GenderTrials,1);
            end
            data{trial}.whichFace(GenderTrials)=whichFace;
            
            for ii=1:3
                Stimulus(:,:,ii)=(double(im{whichFace}(:,:,ii))/255);
            end
            
            if data{trial}.Flipping(GenderTrials) % To flip or not to flip?
                for ii=1:3,Stimulus(:,:,ii)=fliplr(Stimulus(:,:,ii));end
            end
            
            
        end
        
        
        
        
        if trigRespCounter~=1 && GetSecs-RespSecs < data{trial}.max_ISI % || GetSecs-RespSecs>data{trial}.max_rt
            TimeMaxRT=GetSecs;
            lumRatio=.55; % this changes roughly the luminance value of the phase scramble images, which tends to be too high
            StimScramble=(double(imScramble{whichFace})/255).*lumRatio;
            
            [gaussian_L gaussian_R gaussian_M] =MakeFeatureKernel(StimScramble);
        else
            [gaussian_L gaussian_R gaussian_M] =MakeFeatureKernel(Stimulus);
        end
        % Average face as background
        bkgFace_texture=Screen('MakeTexture',windowPtr,(backgroudIm));
        
        % Each feature changing texture (gaussian for smoothed)
        texture_1=Screen('MakeTexture',windowPtr,abs(gaussian_L)*255);
        texture_2=Screen('MakeTexture',windowPtr,abs(gaussian_R)*255);
        texture_3=Screen('MakeTexture',windowPtr,abs(gaussian_M)*255);
        
        
        
        tt = vbl-startSecs;                                                                                                 % time in s from stimulus onset
        for ii = 1:nb_objects,
            sinusoid(ii)= ((sin(data{trial}.freq{ii}(1)*tt*2*pi)/2+0.5))*255;
        end
        
        
        if EEG==1 && trigRespCounter~=1 && GetSecs-RespSecs>data{trial}.max_ISI
            if data{trial}.Gender(GenderTrials) == 1 % which Gender is it (1==Man,0==woman)
                TriggWhichGender=151;
            else
                TriggWhichGender=150;
            end
            
            % %  TRIGGER for Gender (251==man, 250==woman)
            outp(adress,TriggWhichGender)
            WaitSecs(0.004)
            outp(adress,0)
            
        end
        
        
        
        Screen('DrawTexture', windowPtr,bkgFace_texture,[],[xx_center-(expe.xSize/2)  yy_center-(expe.ySize/2) xx_center+(expe.xSize/2) yy_center+(expe.ySize/2)],[]);
        Screen('DrawTexture', windowPtr,  texture_1,[],tags_rect_sc{1},[],[],[],[255 255 255 sinusoid(1)]);
        Screen('DrawTexture', windowPtr,  texture_2,[],tags_rect_sc{2},[],[],[],[255 255 255 sinusoid(2)]);
        Screen('DrawTexture', windowPtr,  texture_3,[],tags_rect_sc{3},[],[],[],[255 255 255 sinusoid(3)]);
        
        
        Screen('DrawLines', windowPtr, fixation_cross_equal, fixation_width, fixation_color, [], 1);
        Screen('DrawLines', windowPtr, fixation_cross_equal, fixation_width, fixation_color, [], 1);
        
        
        % drawing finished
        vbl = Screen('Flip', windowPtr);
        
        
        FlushEvents('keyDown');
        temp = 0;
        whichKey = [];
        
        
        [keyIsDown, secs, keyCode, deltaSecs] = KbCheck;
        temp=KbName(keyCode);
        if keyIsDown && trigRespCounter==1
            trigRespCounter=trigRespCounter+1;
            trigQUEST=1;
            RespSecs=GetSecs;
            
            data{trial}.response{GenderTrials} = temp;
            data{trial}.response_time(GenderTrials,1) = secs-countSecs;
            
            if sum(keyCode)==1 && strcmp(data{trial}.response{GenderTrials}, key_man),               % respond target is present
                if EEG==1
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % %  TRIGGER for man answer (== 251)
                    outp(adress,TriggRespMan)
                    WaitSecs(0.004);
                    outp(adress,0);
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % %  Accuracy
                if data{trial}.Gender(GenderTrials) == 1
                    data{trial}.accuracy(GenderTrials) = 1;
                elseif data{trial}.Gender(GenderTrials) == 0
                    data{trial}.accuracy(GenderTrials) = 0;
                end
                continue;
            elseif sum(keyCode)==1 && strcmp(data{trial}.response{GenderTrials}, key_woman),
                
                if EEG==1
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % %  TRIGGER for Woman answer (== 250)
                    outp(adress,TriggRespWoman)
                    WaitSecs(0.004);
                    outp(adress,0);
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % %  Accuracy
                if data{trial}.Gender(GenderTrials) == 1;
                    data{trial}.accuracy(GenderTrials) = 0;
                elseif data{trial}.Gender(GenderTrials) == 0;
                    data{trial}.accuracy(GenderTrials) = 1;
                end
                continue;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % % Quit Key
            elseif sum(keyCode)==1 && strcmp(data{trial}.response{GenderTrials}, key_quit)
                sca;
                save(file_name, 'data')
                ShowCursor;
                error('Quit key was pressed');
            end
        end
        
        if trigRespCounter~=1 && GetSecs-RespSecs < data{trial}.max_ISI % || GetSecs-RespSecs>data{trial}.max_rt
            if data{trial}.accuracy(GenderTrials)==1
                Screen('DrawText', windowPtr, 'Good!', xx_center-30, yy_center-300, [0,255,100,255]);
            elseif data{trial}.accuracy(GenderTrials)==0
                Screen('DrawText', windowPtr, ' Nope!', xx_center-30, yy_center-300, [255,20,20,255]);
            end
        end
        
        if trigRespCounter~=1 && trigQUEST==1
            trigQUEST=0;
            Questy.q = QuestUpdate(Questy.q, Signal,  data{trial}.accuracy(GenderTrials));
        end
        
        
        
        Screen('Close',texture_1);
        Screen('Close',texture_2);
        Screen('Close',texture_3);
        Screen('Close',bkgFace_texture);
        
        
    end
    % gotta throw out all non-trial data
    data{trial}.Gender((GenderTrials+1):end);
    
    if EEG==1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %  TRIGGER for END of stimulation (== 2)
        outp(adress,TriggENDStim);
        WaitSecs(0.01);
        outp(adress,0);
        % % % % %
    end
    
    
    % Draw larger fixation cross with lines of equal length
    Screen('DrawLines', windowPtr, fixation_cross_equal, fixation_width, fixation_color, [], 1);
    Screen('Flip', windowPtr);
    WaitSecs(blank_target_interval+data{trial}.jitter(3));
    
    
end

% Saves data and closes screen
save(file_name, 'data')
ShowCursor;
sca;

acc_all=[];for trial=1:nb_trialsTot,acc_all=[acc_all; data{trial}.accuracy(:)];end
RT_all=[];for trial=1:nb_trialsTot,RT_all=[RT_all; data{trial}.response_time(:)];end

figure, plot(RT_all), title(sprintf('mean accuracy = %.2f , RT = %1.2f',mean(acc_all(:)),median(RT_all)))

end


function [gaussianLefteye gaussianRighteye gaussianMouth]= MakeFeatureKernel(faceImage)

faceImage=double(faceImage);

if size(faceImage,3)<3 % makes image correct size
    for ii=1:3,faceImage(:,:,ii)=faceImage;end
end

clear gaussian_1 gaussian_2 gaussian_3
border=-10;
eye_hl = 182; % horizontal position of left eye
eye_hr = 316; % horizontal position of right eye


eye_v = 202; % vertical position of both eyes
eye_r = 66; % circular radius of both eyes

mouth_h = 247; % horizontal position of mouth
mouth_v = 339; % vertical position of mouth was 380 for imF from DUpuis-Roy et al.,2009
%     Mouth_v_Flipped=156; % inverted face mouth position

mouth_rh = 100; % horizontal elliptical radius of mouth
mouth_rv = 41; % vertical elliptical radius of mouth


BothEyes=faceImage((eye_v-eye_r)-border:(eye_v+eye_r)+border,:,:);
LeftEye=BothEyes(:,(eye_hl-eye_r)-border:(eye_hl+eye_r)+border,:);
RightEye=BothEyes(:,(eye_hr-eye_r)-border:(eye_hr+eye_r)+border,:);
Mouth=faceImage(:,(mouth_h-mouth_rh)-border:(mouth_h+mouth_rh)+border,:);
Mouth=Mouth((mouth_v-mouth_rv)-border:(mouth_v+mouth_rv)+border,:,:);



% % % % mask preparation first kernel (lefteye) % % % % % %
kernel1_size=(eye_r*2)+(border*2+1);                                                 % can be anything but shouldn't be too small OR too big..
% gaussian_1(:,:,1) = ones(kernel1_size, kernel1_size)
for dim=1:3, gaussian_1(:,:,dim) = ones(kernel1_size, kernel1_size).*squeeze(LeftEye(:,:,dim));end
std_prop = 0.35;                                                             % std of the gaussian in proportion of kernel_size
[x,y] = meshgrid(0:kernel1_size - 1, 0:kernel1_size - 1);
x = x/kernel1_size-.5;
y = y/kernel1_size-.5;
gaussian_1(:,:,4) = exp(-(x .^2 / std_prop ^2) - (y .^2 / std_prop ^2));


% % % mask preparation for second kernel (right eye, same parameter)  % % %
for feat=1:3, gaussian_2(:,:,feat) = ones(kernel1_size, kernel1_size).*squeeze(RightEye(:,:,feat));end
% gaussian_2=mean(gaussian_2,3);
gaussian_2(:,:,4) = exp(-(x .^2 / std_prop ^2) - (y .^2 / std_prop ^2));

% % % mask preparation for third kernel (Mouth)  % % %
kernel3_size_x=mouth_rh*2+(border*2+1);
kernel3_size_y=mouth_rv*2+(border*2+1);
for feat=1:3, gaussian_3(:,:,feat) = ones(kernel3_size_x, kernel3_size_y)'.*squeeze(Mouth(:,:,feat));end
std_prop = 0.35;                                                             % std of the gaussian in proportion of kernel_size
[x,y] = meshgrid(0:kernel3_size_x - 1, 0:kernel3_size_y - 1);
x = x/kernel3_size_x-.5;
y = y/kernel3_size_y-.5;
% gaussian_3=mean(gaussian_3,3);
gaussian_3(:,:,4) = (exp(-(x .^2 / std_prop ^2) - (y .^2 / std_prop ^2)));

gaussianLefteye=gaussian_1;
gaussianRighteye=gaussian_2;
gaussianMouth=gaussian_3;

end


function [Tagged_noise]=scramblePhase(Faceimage)
ratio=.60;
imSize=size(Faceimage);
clear ImScrambled MinNoiseFace
for dim=1:3
    RandomPhase = angle(fft2(rand(imSize(1), imSize(2), 1)));
    
    ImFourier = fft2(Faceimage(:,:,dim));
    %Fast-Fourier transform
    Amp = abs(ImFourier);
    %amplitude spectrum
    Phase = angle(ImFourier);
    %phase spectrum
    Phase2 = Phase + (ratio-1)*RandomPhase;
    PhaseMinNoise=Phase + (0.25-1)*RandomPhase;
    %add random phase to original phase
    temp_scramble = ifft2(Amp.*exp(sqrt(-1)*(Phase2)));
    ImMinNoise= ifft2(Amp.*exp(sqrt(-1)*(PhaseMinNoise)));
    %combine Amp and Phase then perform inverse Fourier
    
    
    
    ImScrambled(:,:,dim) = uint8(real([temp_scramble+ImMinNoise]./2)); %get rid of imaginery
    MinNoiseFace(:,:,dim)=uint8(real(ImMinNoise));
end
Tagged_noise=ImScrambled;
Tagged_noise=squeeze(MinNoiseFace);


end


function tim = noisy_bit(im, depth)
% im must vary between 0 and 1 and depth is the number of gray shades
%
% im = double(imread('w1N.JPG'))/255;
% figure, imshow(stretch(noisy_bit(im, 2)))
%
% Allard, R., Faubert, J. (2008) The noisy-bit method for digital displays:
% converting a 256 luminance resolution into a continuous resolution.
% Behavior Research Method, 40(3), 735-743.
%
% Frederic Gosselin, 12/02/2013
% frederic.gosselin@umontreal.ca

tim = im*(depth-1);
tim = max(min(round(tim+rand(size(im))-.5), depth-1), 0) + 1;
end
