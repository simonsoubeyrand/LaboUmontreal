
function XpInstruction(fileName,format)
% %  XpInstruction displays a given format file on Screen until keypress
%
%   Author:     Simon Faghel-Soubeyrand
%   Date:       October/2015
%   Version:    1
%   Tested by:  Frederic Gosselin

Screen('Preference', 'SkipSyncTests', 1); % uncomment if necessary

% Psychophysics inits
AssertOpenGL;
screens=Screen('Screens');
screenNumber=max(screens);
[w, wRect]=Screen('OpenWindow',screenNumber, 128,[],32,2);

%  Center coordinates
[xCenter,yCenter]   = RectCenter(wRect);

%
% % Open window with default settings:
% w=Screen('OpenWindow', screenNumber,128);
% Select specific text font, style and size:
Screen('TextFont',w, 'Helvetica');
Screen('TextSize',w, 23);
Screen('TextStyle', w, 1);


% Read some text file, if existing. .m are prefered...
fid = fopen(sprintf('%s.%s', fileName,format),'r');

if fid==-1
    error('Could not open text file: it appears inexistant or unavailable');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read every line of the file and put it in "text" string
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
text = '';
tl = fgets(fid);
lcount = 0;
try
    while 1
        
        if ~ischar(tl), break, end
        if(~isempty(line)&&line~=-1)
            text = [text tl];
            tl = fgets(fid);
            % tl=textscan(tl,'%s');
            lcount = lcount + 1;
        end
    end
catch e
    e.getReport
    ttt
    keyboard
end

fclose(fid);
text = [text newline];

% Draw centered text inside frame
[nx, ny, bbox] = DrawFormattedText(w, text, 'center', 'center', 0);

Screen('FrameRect', w, [10 10 10], bbox+[-20 -20 +20 +20],3);

Screen('DrawText', w, 'Appuyez une touche sur le clavier pour débuter', xCenter-250, ny+50, [0, 100, 100, 255]);
Screen('Flip',w);
WaitSecs(1);
clear Screen screens
KbWait;
WaitSecs(.5);


end