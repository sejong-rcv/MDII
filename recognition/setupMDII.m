function imdb = setupMDII(trainType, numTrain, numTest, datasetDir, varargin)
% SETUPCALTECH256    Setup Caltech 256 and 101 datasets
%    This is similar to SETUPGENERIC(), with modifications to setup
%    Caltech-101 and Caltech-256 according to the standard
%    evaluation protocols. Specific options include:
%
%    Variant:: 'caltech256'
%      Either 'caltech101' or 'caltech256'.
%
%    AutoDownload:: true
%      Automatically download the data from the Internet if not
%      found at DATASETDIR.
%
%    See:: SETUPGENERIC().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.lite = false ;
opts.numTrain = numTrain ;
opts.numTest = numTest ;
opts.seed = 1 ;
opts.variant = 'MDII' ;
opts.autoDownload = false ;
opts = vl_argparse(opts, varargin) ;
numClasses = 1 % RGB base, Thermal base
vl_xmkdir(datasetDir) ;

% Read classes
imdb = setupGenericMDII(trainType, datasetDir, ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', numClasses, ...
  'seed', opts.seed, 'lite', opts.lite) ;

switch opts.variant
  case 'caltech256'
    imdb.images.set(imdb.images.class == 257) = 0 ;
    ok = find(imdb.images.set ~= 0) ;
    imdb.images.id = imdb.images.id(ok) ;
    imdb.images.name = imdb.images.name(ok) ;
    imdb.images.set = imdb.images.set(ok) ;
    imdb.images.class = imdb.images.class(ok) ;
end

