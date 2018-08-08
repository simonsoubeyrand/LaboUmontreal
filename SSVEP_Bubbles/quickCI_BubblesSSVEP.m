function quickCI_BubblesSSVEP(name,task,font)


% load('/Users/SimonFaghel-Soubeyrand/Desktop/TESTrose/Bubbles_SSVEP_rosetest_gender_1.mat')

sizeX= 128;
masque2D=double(imread('/Users/SimonFaghel-Soubeyrand/Desktop/basic Stat4Ci tools/masque.tif'));
masque=masque2D(:);
%back = mean(double(imread('mF_91')),3)/255;
nSubjects=1;
nBins=10;
nTrials=100;
nTrialsPerBin   = nTrials/nBins;
thr             = .75; %Threshold for bubbles computation
sizeX=128;


p = .05;
tC = 2.7;
counter=0;
sigma=8;
counter=counter+1;
FWHM(counter) = sigma * sqrt(8*log(2));

[volumes, N] = CiVol(sum(masque(:)), 2);
[tP, k,tC1] = stat_threshold(volumes, N, FWHM, Inf, p, tC);  % the actual statistical tests
tP


n=0;
h = waitbar(0, 'ICs : 0 % compl�t�');

CI_RTall=zeros(128);

load 'images.mat'; % load images

% Bulles
sigma_bulles       = 3;
TNoyau      = 6*sigma;
bulle       = fspecial('gaussian',ceil(TNoyau),sigma_bulles);
bulle       = bulle - min(bulle(:));
bulle       = bulle /sum(bulle(:));

for subject = 1
    X_all = [];
    y_all = [];
    RT_all = [];
    CID.DATA = [];
    contrast = [];
    
    n=n+1;
    
    
    
    for block = 1:2,
        
        
        
        fname=sprintf('/Users/SimonFaghel-Soubeyrand/Desktop/TESTrose/Bubbles_SSVEP_%s_%s_%d.mat',name,task,block);
        load(fname); %e.g :
        %cid.data=cid.data;
        cid.DATA    = [CID.DATA cid.data];
        bubbles_block= mean(cid.data(5,:));
        %
        avrBub  = zeros(1,nBins);
        avrAcc  = zeros(1,nBins);
        for nbin = 1:nBins
            nbBubbles    = cid.data(5,(nbin-1)*nTrialsPerBin+1:nbin*nTrialsPerBin);
            avrBub(nbin) = mean(nbBubbles);
            acc          = cid.data(9,(nbin-1)*nTrialsPerBin+1:nbin*nTrialsPerBin);
            avrAcc(nbin) = mean(acc);
            
        end
        
        
        
        bubbles=cid.data(5,:);
        
        % Initialize vectors and matrices
        nTrials     = size(cid.data, 2);
        X           = zeros(nTrials, sizeX^2);
        y           = zeros(1, nTrials);
        
        % Get the seed from the cid and initialize the rand function
        temp        = sscanf(cid.noise, '%s%s%s%s%d');
        seed_0      = temp(end);
        rand('state', seed_0); %initializing
        
        % Reproduce the noise for each trial and put it in a matrix
        for trial = 1:nTrials
            
            
            % Creation de bruit
            qteBulles = cid.data(5,trial);
            
            %mask=zeros(sizeX,sizeX,3);
            prob_tmp = qteBulles/sum(masque(:));
            tmp=rand(sizeX^2,1) .* masque(:);% was .*masque
            tmp2=reshape(tmp>=(1-prob_tmp),sizeX,sizeX); % makes the criteria probabilistic
            masque2D=filter2(bulle,tmp2);
            masque2D = (masque2D - min(masque2D(:)))/(max(masque2D(:)) - min(masque2D(:)));
            % mask= repmat(masque2D,[1 1 3]);% was repmat(masque2D, 1, 1, 3)
            tmp = tmp2;
            X(trial,:) = tmp(:);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%
%             X(trial,:) = (X(trial,:)-mean(X(trial,:)))/std(X(trial,:));
            %%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Variable dependente
            y(1,trial)  = cid.data(9,trial); % ACCURACY
            RT(1,trial)  = cid.data(8,trial); % Response Time
            
            
            if find(isnan(X(trial,:)))
                X(trial,:)=zeros(1,sizeX*sizeX);
                %%%%%%%%%%%%%%
                % SOME PARTICIPANTS HAD 1 OR 0 BUBBLES,  made NAN WHEN WE DIVISE BY 0
                %%%%%%%%%%%%%%%
            end
            
        end
        %         makeCI(y,X,8);
        
        % Standardisation de la VD
        y       = (y - mean(y)) / std(y);
        RT       = -1 * ((RT - mean(RT)) / std(RT));
        
        X_all = [X_all; X;];
        y_all = [y_all y];
        
        RT_all = [RT_all RT];
        
        
        b_part = y * X;
        b_2D = reshape(b_part, sizeX, sizeX);
        
        
        % Bootstrapping the accuracry vector.
        %index = randperm(size(y,2)); % sans remise
        index = ceil(size(y,2)*rand(size(y))); % avec remise, preferable
        b_boot = y(index) * X;
        b_boot_2D = reshape(b_boot, sizeX, sizeX);
        
        
        
        % same with RT.
        b_RT= RT * X;
        b_2D_RT=reshape(b_RT, sizeX, sizeX);
        
        index = ceil(size(RT,2)*rand(size(RT))); % avec remise, preferable
        b_bootRT= RT(index) * X;
        b_bootRT=reshape(b_bootRT, sizeX, sizeX);
        
        
        
        left = sum(left_target(:).*b_2D(:));
        right = sum(right_target(:).*b_2D(:));
        
        contrast(n,block) = (left-right)/(left+right);
        
        
        %   figure, imagesc(SmoothCi(b_boot_2D,sigma)), title(sprintf(' %s cond %d, CONTRAST :%f', name, condition, mean(contrast(:)))), colorbar
    end
    
    
    
    %On combine les blocs pour chq sujets..
    %     SSP=makeIC(X_all,y_all,sizeX,sigma);
    %
    %
    b_all = y_all * X_all;
    b_2D_all = reshape(b_all, sizeX, sizeX);
    
    % Bootstrapping the accuracry vector.
    index = randperm(size(y_all,2));
    b_boot = y_all(index) * X_all;
    b_boot_2D_all = reshape(b_boot, sizeX, sizeX);
    
    ci = SmoothCi(b_2D_all, sigma);
    ci_boot = SmoothCi(b_boot_2D_all, sigma);
    
    
    goodRTindex=RT_all>-4;
    b_RT_all= RT_all * X_all;
    b_2D_RT_all=reshape(b_RT_all, sizeX, sizeX);
    
    
    %   figure, imagesc(SmoothCi(b_2D_RT,8)),colorbar
    
    
    CI_RTall=CI_RTall+ SmoothCi(b_2D_RT_all,sigma);
    
    
    % Standardize sci with sci_boot
    SSP = (ci - mean(ci_boot(:))) / std(ci_boot(:));
    
    temp=SSP.*masque2D;
    sumright=mean(sum(temp(:,65:128)));
    sumleft=mean(sum(temp(:,1:64)));
    back=double(imF{1}(:,:,1));
    
    
    
    if font==1
        [out threshold wtv] = overlay_cluster(back,SSP,[tC],k);
        figure, imagesc(out), title(sprintf(' %s, %s, left: %3.2f, right :%3.2f', name, task,sumleft,sumright))
%         hold on
%         contour(SSP>tC,'-k')
    else
        figure, imagesc(SSP), title(sprintf(' %s, %s,left: %3.2f, right :%3.2f', name,task ,sumleft,sumright)),colorbar,colormap('jet')
    end
    
    waitbar(block/3, h, sprintf('Image de Classification %s : %3.2f %% compl�t�', name,block/3));
    
    
    
end
delete(h)
% figure, imagesc(CI_RTall),colorbar