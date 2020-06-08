function recognition_demo(trainType, testType, MDII_numTrain, MDII_numTest, varargin)
% RECOGNITION_DEMO  Demonstrates using VLFeat for image classification

if ~exist('vl_version')
  run(fullfile(fileparts(which(mfilename)), ...
               '..', '..', 'toolbox', 'vl_setup.m')) ;
end

opts.dataset = 'caltech101' ;
opts.prefix = 'bovw' ;
opts.encoderParams = {'type', 'bovw'} ;
opts.seed = 1 ;
opts.lite = true ;
opts.C = 1 ;
opts.kernel = 'linear' ;
opts.dataDir = 'data';
for pass = 1:2
  opts.datasetDir = fullfile(opts.dataDir, opts.dataset) ;
  opts.resultDir = fullfile(opts.dataDir, opts.prefix) ;
  opts.imdbPath = fullfile(opts.resultDir, 'imdb.mat') ;
  opts.imdbTestPath = fullfile(opts.resultDir, 'imdbTest.mat') ;
  opts.encoderPath = fullfile(opts.resultDir, 'encoder.mat') ;
  opts.modelPath = fullfile(opts.resultDir, 'model.mat') ;
  opts.diaryPath = fullfile(opts.resultDir, 'diary.txt') ;
  opts.cacheDir = fullfile(opts.resultDir, 'cache') ;
  opts = vl_argparse(opts,varargin) ;
end

% do not do anything if the result data already exist
if exist(fullfile(opts.resultDir,'result.mat')),
  load(fullfile(opts.resultDir,'result.mat'), 'ap', 'confusion') ;
  fprintf('%35s mAP = %04.1f, mean acc = %04.1f\n', opts.prefix, ...
          100*mean(ap), 100*mean(diag(confusion))) ;
  return ;
end

vl_xmkdir(opts.cacheDir) ;
diary(opts.diaryPath) ; diary on ;
disp('options:' ); disp(opts) ;

% --------------------------------------------------------------------
%                                                   Get image database
% --------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath);
else
 switch opts.dataset
   case 'MDII'
       datasetDir=[opts.datasetDir '/train'];
       imdb = setupMDII(trainType, MDII_numTrain, MDII_numTest, datasetDir, 'lite', opts.lite) ;
   otherwise, error('Unknown dataset type.') ;
 end
 save(opts.imdbPath, '-struct', 'imdb') ;
end

if exist(opts.imdbTestPath)
  imdbtest = load(opts.imdbTestPath);
else
 datasetTestDir=[opts.datasetDir '/test'];
 imdbtest = setupMDII(testType, MDII_numTrain, MDII_numTest, datasetTestDir, 'lite', opts.lite) ;
 save(opts.imdbTestPath, '-struct', 'imdbtest') ;
end


% --------------------------------------------------------------------
%                                      Train encoder and encode images
% --------------------------------------------------------------------

if exist(opts.encoderPath)
  encoder = load(opts.encoderPath) ;
else
  numTrain = 5000 ;
  if opts.lite, numTrain = 10 ; end
  train = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform') ;
  encoder = trainEncoder(fullfile(imdb.imageDir,imdb.images.name(train)), ...
                         opts.encoderParams{:}, ...
                         'lite', opts.lite) ;
  save(opts.encoderPath, '-struct', 'encoder') ;
  diary off ;
  diary on ;
end

if exist(fullfile(opts.resultDir,'descrstrain.mat'))
  descrs = load(fullfile(opts.resultDir,'descrstrain.mat')) ;
else
  descrs = encodeImage(encoder, fullfile(imdb.imageDir, imdb.images.name), ...
  'cacheDir', opts.cacheDir) ;
  save(fullfile(opts.resultDir,'descrstrain.mat') , 'descrs') ;
end

if exist(fullfile(opts.resultDir,'descrstest.mat'))
  descrs = load(fullfile(opts.resultDir,'descrstest.mat')) ;
else
  mkdir( fullfile(opts.cacheDir,'test'))
  descrstest = encodeImage(encoder, fullfile(imdbtest.imageDir, imdbtest.images.name), ...
'cacheDir', fullfile(opts.cacheDir,'test')) ;
  save(fullfile(opts.resultDir,'descrstest.mat') , 'descrstest') ;
end

diary off ;
diary on ;

end
