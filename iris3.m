%--------------------------------------
% CSCI 59000 Biometrics - Iris/Retina Extraction 
% Author: Chu-An Tsai
% 02/23/2020
%--------------------------------------

% This code is to detect the circles of pupil and iris, and find the boundary of eyelids.
% Then take the intersection of the donut region and detected eyelid region.
% Finally, unwrapping the donut.
% For iris3.jpg

clear,clc;

image = imread('images/iris3.jpg');
image = rgb2gray(image);
[h,w] = size(image);
figure(1);
set (gcf,'Position',[0,300,w,h]);
%imshow(image);

% Find the inner circle(pupil) and the outer circle(iris)
[centers_inner, radii_inner, metric_inner] = imfindcircles(image,[30 50], 'ObjectPolarity','dark','Sensitivity',0.95,'EdgeThreshold',0.1);
[centers_outer, radii_outer, metric_outer] = imfindcircles(image,[90 120],'ObjectPolarity','dark','Sensitivity',0.95,'EdgeThreshold',0.05);

subplot(231)
imshow(image);
hold on
viscircles(centers_inner, radii_inner, 'EdgeColor','r');

subplot(232)
imshow(image);
hold on
viscircles(centers_outer, radii_outer, 'EdgeColor','b');

subplot(233)
imshow(image);
hold on
viscircles(centers_inner, radii_inner, 'EdgeColor','r');
hold on
viscircles(centers_outer, radii_outer, 'EdgeColor','b');

% Compute binary masks
bim_inner = zeros(h,w);
bim_outer = zeros(h,w);
xy_inner = [centers_inner(2),centers_inner(1)];
xy_outer = [centers_outer(2),centers_outer(1)];

for i = 1 : h
    for j = 1 : w
        xy = [i,j];
        if norm(xy-xy_inner) <= radii_inner 
            bim_inner(i,j) = 1;
        end
        if norm(xy-xy_outer) <= radii_outer
            bim_outer(i,j) = 1;
        end
    end
end

subplot(234)
imshow(bim_inner)
%hold on
%viscircles(centers_inner, radii_inner, 'EdgeColor','r');

subplot(235)
imshow(bim_outer)
%hold on
%viscircles(centers_outer, radii_outer, 'EdgeColor','b');

% Donut mask
bim_donut = bim_outer - bim_inner;

subplot(236)
imshow(bim_donut)
%hold on
%viscircles(centers_inner, radii_inner, 'EdgeColor','r');
%hold on
%viscircles(centers_outer, radii_outer, 'EdgeColor','b');

% Using snake function to detect upper and lower eyelids
[Px, Py] = snake_pgm('images/iris3.jpg');

% Put the first point back to make the graph completely connected
Px = [Px.',Px(1)];
Py = [Py.',Py(1)];

figure();
temp = zeros(h,w);
imshow(temp,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,w,h]);
axis normal;
hold on
plot(Px,Py,'-g','LineWidth',2);
% Export the graph to get the outline
saveas(gcf,'temp.bmp');

figure();
imshow(image)
hold on
viscircles(centers_inner, radii_inner, 'EdgeColor','r');
hold on
viscircles(centers_outer, radii_outer, 'EdgeColor','b');
hold on
plot(Px,Py,'-g','LineWidth',2);

% Read the graph and make the eyelid region a mask
image_outline = rgb2gray(imread('temp.bmp'));
image_eyelids = imresize(image_outline,[h w]);
eyelid_region = imbinarize(image_eyelids);
figure();
subplot(131)
imshow(eyelid_region)
bim_eyelid = imfill(eyelid_region,'holes');
subplot(132)
imshow(bim_eyelid)
%hold on
%plot(Px,Py,'-g','LineWidth',2);

% Take the intersection of donut mask and eyelid region mask
final_bim_donut = and(bim_donut,bim_eyelid);
subplot(133)
imshow(final_bim_donut)

% Get the real donut from the original image with a white background
donut = image;
count1 = 0;
for i = 1 : h
    for j = 1 : w
        if final_bim_donut(i,j) ~= 1
            donut(i,j) = 255;
        else
            count1 = count1 + 1;
        end
    end
end

figure();
imshow(donut)

% Unwrap the donut
inner_boundary = ceil(radii_inner*0.97);  
outer_boundary = ceil(radii_inner*2.65);
rows = ceil(outer_boundary-inner_boundary);
columns = ceil(pi*(outer_boundary+inner_boundary));
al = 2*pi/columns;
for i = 1:columns
    point_A_x(1,i) = centers_inner(2) + inner_boundary*cos(i*al);
    point_A_y(1,i) = centers_inner(1) + inner_boundary*sin(i*al); 
    point_B_x(1,i) = centers_inner(2) + outer_boundary*cos(i*al);
    point_B_y(1,i) = centers_inner(1) + outer_boundary*sin(i*al); 
end
for i = 1:rows
    point_x(i,:) = round(point_A_x(1,:) + (point_B_x(1,:) - point_A_x(1,:))*i/rows) ; 
    point_y(i,:) = round(point_A_y(1,:) + (point_B_y(1,:) - point_A_y(1,:))*i/rows) ;
end
for i = 1:rows      
    for j = 1:columns
        unwrapped_donut(i,j) = donut(point_x(i,j),point_y(i,j));
    end
end

figure();
imshow(unwrapped_donut)

count2 = 0;
for i = 1 : rows
    for j = 1 : columns
        if unwrapped_donut(i,j)~= 255
            count2 = count2 + 1;
        end
    end
end

if count1 <= count2
    fprintf('You''ve got all points included!')
else
    disp(count1)
    disp(count2)
end

%--------- Function used (from "snake.zip" with some revise)---------------

function [Px, Py] = snake_pgm(filename)

[I,map] = imread(filename);  
I = rgb2gray(I);
% Compute its edge map, 
f = I/255;
f0 = gaussianBlur(f,2);
[px,py] = gradient(f0);

% get the edge strength image
e = sqrt(px.^2 + py.^2);
[px, py] = gradient(e);

figure; 
imdisp(f0); 

% now allow manually input the snake points
[x,y] = snakeinit(1.0);	
[x,y] = snakeinterp1(x,y,3); % snakeinterp1 does interpolation based on arc length

% snake deformation
disp(' ');
disp(' Press any key to start the deformation');
pause;

for i=1:200
   [x,y] = snakedeform(x,y,0.05,0.005,1,4,px,py,5);
   [x,y] = snakeinterp1(x,y,3); % reparamerization
   snakedisp(x,y,'r'),
   title(['Deformation in progress,  iter = ' num2str(i*5)])
   pause(0.00001);
end

disp(' ');
disp(' Press any key to display the final result');
pause;

figure; 
imdisp(f0); 
snakedisp(x,y,'r') 
title(['Final result,  iter = ' num2str(i*5)]);

but =1;

while(1)
   
   disp(' ');
   disp('If satisfied with the final result,Press middle mouse button to exit');
   disp(' ');
   
   disp('If not satisfied with the final result, pull the snake, please');
   disp('Click the left mouse button to pick points to move toward.')
   disp('Press the right mouse button to continue the deformation.')
   
   % If only right mouse button clicked, then the snake will simply do more iterations
   % of convergence.
   Ptx = [];
   Pty = [];
   n =0;
   
   [s, t, but] = ginput(1);
   
   % If this is the middle button, exit
   if(but==2)
      break;
   end; 
   
   maxGradient = max(e(:));
   ee =e;
   
   hold on;
   % but =1, picked the first point
   while but == 1
      n = n + 1;
      Ptx(n,1) = s;
      Pty(n,1) = t;
      plot(Ptx, Pty, 'b*');
      s = round(s);
      t = round(t);
      ee(s-2:s+2,t-2:t+2) = maxGradient;
      
      % pick another point (left button) or start deformation (right button) 
      [s, t, but] = ginput(1);
   end   
   
   if(but==2)
      break;
   end;
   
   [ppx, ppy] = gradient(ee);
   
   % but=3, start deformation   
   for i=1:50,
      
      % Every attraction point's closest snake point will feel a force
      for j = 1:size(Ptx, 1)
         s =Ptx(j, 1);
         t =Pty(j, 1);
         
         d = sqrt((x-s).^2 + (y-t).^2);
         minD = min(d);
         idx = find(d==minD);
                  
         x(idx) =s;
         y(idx) =t;
      end;
      
      [x,y] = snakedeform(x,y,0.05,0.03,1,4,ppx,ppy,5);
      [x,y] = snakeinterp1(x,y,3); 
      snakedisp(x,y,'r') 
      title(['Deformation in progress,  iter = ' num2str(i*5)])
      pause(0.00001);
   end
   
   %show final result
   snakedisp(x,y,'g');
   
   disp(' ');
   disp(' Press any key to display the final result');
   pause;
   
   figure,
   imdisp(f0); 
   snakedisp(x,y,'r') 
end;   

% return the snake points
Px =x;
Py =y;
end
function GI = gaussianBlur(I,s)
% GAUSSIANBLUR blur the image with a gaussian kernel
%     GI = gaussianBlur(I,s) 
%     I is the image, s is the standard deviation of the gaussian
%     kernel, and GI is the gaussian blurred image.


M = gaussianMask(1,s);
M = M/sum(sum(M));   % normalize the gaussian mask so that the sum is
                     % equal to 1
GI = xconv2(I,M);
end
function M = gaussianMask(k,s)
% k: the scaling factor
% s: standard variance

R = ceil(3*s); % cutoff radius of the gaussian kernal  
for i = -R:R,
    for j = -R:R,
        M(i+ R+1,j+R+1) = k * exp(-(i*i+j*j)/2/s/s)/(2*pi*s*s);
    end
end

end
function Y = xconv2(I,G)
% function Y = xconv2(I,G)
%   I: the original image
%   G: the mask to be convoluted
%   Y: the convoluted result (by taking fft2, multiply and ifft2)
% 
%   a similar version of the MATLAB conv2(I,G,'same'),  7/10/95
%   implemented by fft instead of doing direct convolution as in conv2
%   the result is almost same , differences are under 1e-10.
%   However, the speed of xconv2 is much faster than conv2 when
%   gaussian kernel has large standard variation.

[n,m] = size(I);
[n1,m1] = size(G);
FI = fft2(I,n+n1-1,m+m1-1);  % avoid aliasing
FG = fft2(G,n+n1-1,m+m1-1);
FY = FI.*FG;
YT = real(ifft2(FY));
nl = floor(n1/2);
ml = floor(m1/2);
Y = YT(1+nl:n+nl,1+ml:m+ml);
end
function imdisp(I)
% imdisp(I) - scale the dynamic range of an image and display it.

x = (0:255)./255;
grey = [x;x;x]';
minI = min(min(I));
maxI = max(max(I));
I = (I-minI)/(maxI-minI)*255;
image(I);
axis('square','off');
colormap(grey);

end
function snakedisp(x,y,style)
% SNAKEDISP  Initialize the snake 
%      snakedisp(x,y,line)
%       
%      style is same as the string for plot

hold on

% convert to column data
x = x(:); y = y(:);

if nargin == 3
   plot([x;x(1,1)],[y;y(1,1)],style);
   hold off
else
   disp('snakedisp.m: The input parameter is not correct!'); 
end
end
function X = uppertri(M,N)
% UPPERTRI   Upper triagonal matrix 
%            UPPER(M,N) is a M-by-N triagonal matrix

[J,I] = meshgrid(1:M,1:N);
X = (J>=I);
end
function [x,y] = snakeinit(delta)
%SNAKEINIT  Manually initialize a 2-D, closed snake 
%   [x,y] = SNAKEINIT(delta)
%
%   delta: interpolation step

hold on

x = [];
y = [];
n =0;

% Loop, picking up the points
disp('Left mouse button picks points.')
disp('Right mouse button picks last point.')

but = 1;
while but == 1
      [s, t, but] = ginput(1);
      n = n + 1;
      x(n,1) = s;
      y(n,1) = t;
      plot(x, y, 'r-');
end   

plot([x;x(1,1)],[y;y(1,1)],'r-');
hold off

% sampling and record number to N
x = [x;x(1,1)];
y = [y;y(1,1)];
t = 1:n+1;
ts = [1:delta:n+1]';
xi = interp1(t,x,ts);
yi = interp1(t,y,ts);
n = length(xi);
x = xi(1:n-1);
y = yi(1:n-1);
end
function [xi,yi] = snakeinterp1(x,y,RES)
% SNAKEINTERP1  Interpolate the snake to have equal distance RES
%     [xi,yi] = snakeinterp(x,y,RES)
%
%     RES: resolution desired

%     update on snakeinterp after finding a bug

% convert to column vector

%x = x(:); y = y(:);

N = length(x);  

% make it a circular list since we are dealing with closed contour

x = [x;x(1)];
y = [y;y(1)];

dx = x([2:N+1])- x(1:N);
dy = y([2:N+1])- y(1:N);
d = sqrt(dx.*dx+dy.*dy);  % compute the distance from previous node for point 2:N+1

d = [0;d];   % point 1 to point 1 is 0 

% now compute the arc length of all the points to point 1
% we use matrix multiply to achieve summing 
M = length(d);
d = (d'*uppertri(M,M))';

% now ready to reparametrize the closed curve in terms of arc length
maxd = d(M);

if (maxd/RES<3)
   error('RES too big compare to the length of original curve');
end

di = 0:RES:maxd;

xi = interp1(d,x,di);
yi = interp1(d,y,di);

N = length(xi);

if (maxd - di(length(di)) <RES/2)  % deal with end boundary condition
   xi = xi(1:N-1);
   yi = yi(1:N-1);
end

xi = xi';
yi = yi';
end
function [x,y] = snakedeform(x,y,alpha,beta,gamma,kappa,fx,fy,ITER)
% SNAKEDEFORM  Deform snake in the given external force field
%     [x,y] = snakedeform(x,y,alpha,beta,gamma,kappa,fx,fy,ITER)
%
%     alpha:   elasticity parameter
%     beta:    rigidity parameter
%     gamma:   viscosity parameter
%     kappa:   external force weight
%     fx,fy:   external force field

% generates the parameters for snake

N = length(x);

alpha = alpha* ones(1,N); 
beta = beta*ones(1,N);

% produce the five diagnal vectors
alpham1 = [alpha(2:N) alpha(1)];
alphap1 = [alpha(N) alpha(1:N-1)];
betam1 = [beta(2:N) beta(1)];
betap1 = [beta(N) beta(1:N-1)];

a = betam1;
b = -alpha - 2*beta - 2*betam1;
c = alpha + alphap1 +betam1 + 4*beta + betap1;
d = -alphap1 - 2*beta - 2*betap1;
e = betap1;

% generate the parameters matrix
% based on the equation (3.20), and set the space size h=1
A = diag(a(1:N-2),-2) + diag(a(N-1:N),N-2);
A = A + diag(b(1:N-1),-1) + diag(b(N), N-1);
A = A + diag(c);
A = A + diag(d(1:N-1),1) + diag(d(N),-(N-1));
A = A + diag(e(1:N-2),2) + diag(e(N-1:N),-(N-2));

invAI = inv(A + gamma * diag(ones(1,N)));

for count = 1:ITER,
   vfx = interp2(fx,x,y,'*linear');
   vfy = interp2(fy,x,y,'*linear');

   % deform snake
   x = invAI * (gamma* x + kappa*vfx);
   y = invAI * (gamma* y + kappa*vfy);
end
end

%--------------------------------------------------------------------------
