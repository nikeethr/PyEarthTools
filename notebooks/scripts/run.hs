#! /usr/bin/env nix-shell
#! nix-shell -i runghc --packages ghcid "ghc.withPackages (x: [ x.turtle x.text ])"

{-# LANGUAGE OverloadedStrings #-}

import Control.Monad (mapM)
import Data.List (isPrefixOf)
import Data.Maybe
import qualified Data.Text as T
import Turtle
import Prelude hiding (FilePath)

parser :: Parser (FilePath, Maybe FilePath, Maybe Int, Maybe Int, Maybe Int)
parser =
    (,,,,)
        <$> argPath "era5_path" "Local path to data - will download if doesn't exist"
        <*> optional (optPath "work_dir" 't' "work dir; default: $TMPDIR or $PBS_JOBFS")
        <*> optional (optInt "epochs" 'e' "No. epochs; default: 2")
        <*> optional
            (optInt "batch_size" 'b' "batch size - depends on memory; default: 32")
        <*> optional
            ( optInt
                "num_workers"
                'w'
                "workers - depends on num gpus; default: ~ num_cpu / num_gpu"
            )

-- TODO:
-- resolveData :: FilePath -> ()

-- TODO:
-- run command

-- TODO:
-- check exists
-- otherwise make directories
resolveWorkDir :: Maybe FilePath -> Maybe FilePath -> FilePath
resolveWorkDir _wd _jobfs =
    to_era5dir $ case (_wd, _jobfs) of
        (Just _w', _) -> _w'
        (Nothing, Just _j') -> _j'
        (Nothing, Nothing) -> "/tmp"
  where
    to_era5dir _r = _r </> "era5lowres"

-- Sets the following from command line:
-- ERA5MINI__NUM_WORKERS
-- ERA5MINI__BATCH_SIZE
-- ERA5MINI__EPOCHS
-- , Additionally sets defaultEnvVars:
setEnvVars :: (MonadIO io) => [(String, String)] -> io ()
setEnvVars es =
    mapM_ (\(_k, _v) -> export (T.pack _k) (T.pack _v)) (defaultEnvVars ++ es)

defaultEnvVars :: [(String, String)]
defaultEnvVars =
    [ ("DASK_DISTRIBUTED__COMM__UCX__INFINIBAND", "True")
    , ("DASK_DISTRIBUTED__COMM__UCX__RDMACM", "True")
    , ("DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT", "True")
    , ("DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY", "True")
    , ("DASK_DISTRIBUTED__COMM__UCX__TCP", "True")
    , ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    , ("OMP_NUM_THREADS", "1")
    , ("MKL_NUM_THREADS", "1")
    , ("OPENBLAS_NUM_THREADS", "1")
    ]

main = do
    (_d, _wd, _e, _b, _w) <-
        options "Runs mini era5lowres tutorial in a script" parser

    -- TODO:
    -- resolveData

    _maybeJobFs <- need "PBS_JOBFS"

    let _maybeJobFsStr = (fmap T.unpack _maybeJobFs)

    -- TODO
    -- ("ERA5MINI__DATA_DIR", resolveDataDir _d)
    setEnvVars
        [ ("ERA5MINI__WORK_DIR", resolveWorkDir _wd _maybeJobFsStr)
        , ("ERA5MINI__NUM_WORKERS", show (fromMaybe 8 _w))
        , ("ERA5MINI__BATCH_SIZE", show (fromMaybe 32 _b))
        , ("ERA5MINI__EPOCHS", show (fromMaybe 2 _e))
        ]

    envCheck <- env
    print $ filter (\(_k, _v) -> isPrefixOf "ERA5" (T.unpack _k)) envCheck
