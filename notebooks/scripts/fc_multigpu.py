# ----------------------------------------------------------------------------
# Multi GPU version of FourcastNeXtMini Tutorial. Slight modification to
# trainer and dataset required in order to partition (or replicate) things
# across GPUs.
#
# torch.DistributedDataPipelines (DDP) is preferable as it invokes
# multiprocessing rather than multithreading which is flakey (old way of doing
# it - torch.DataParallel)
#
# References
#   * TODO
#
# Author: Nikeeth R. (nikeeth.ramanathan@bom.gov.au)
# ----------------------------------------------------------------------------

# torch wraps multiprocessing with some magic We should really compare with
# pure torch v.s. lightning:
# - https://github.com/Lightning-AI/pytorch-lightning/discussions/10382
# - https://stackoverflow.com/questions/78933748/why-is-pytorch-lightning-slower-than-my-basic-pytorch-code
# - https://medium.com/@florian-ernst/finding-why-pytorch-lightning-made-my-training-4x-slower-ae64a4720bd1
# - https://github.com/Lightning-AI/pytorch-lightning/issues/6196
# - https://www.reddit.com/r/deeplearning/comments/t31ppy/for_what_reason_do_you_or_dont_you_use_pytorch/
# - "lightning is built by researchers for researchers" (why do we need PET then?)
#
# Thoughts:
# - If we are making our own framework lightning adds too many extra layers of
#   indirection and side effects...
# - Its good for researchers to do research, without questioning the insides.
#
# Changes:
# - import declarations are scoped
# - explicit setting of forkserver
# - using torch.multiprocessing mutex sprinkle in places (not here internal)
# - caching in places (not here - internal)

import torch
import torch.multiprocessing as mp


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def get_workdir():
    _workdir = os.environ["ERA5LOWRESDEMO"]
    os.makedirs(_workdir, exist_ok=True)
    return _workdir

def get_data_minime():
    return get_workdir() + "/mini.nc"

def clear_gpu_cache_and_stale_things():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------------
def download_mini_dataset():
    # NOTE: change this if needed
    import xarray as xr
    import os
    file_location = get_data_minime()
    if not os.path.exists(file_location):
        print("Training data not found, downloading around 2.8GB of data")
        era5_lowres = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr")
        subset = era5_lowres[[
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]]
        subset.to_netcdf(file_location, engine="netcdf4")
        assert os.path.exists(file_location)
        print("Wrote file to {file_location}")
    else:
        print("File already downloaded, skipping ...")

# ----------------------------------------------------------------------------
# SETUP - PET/lightning things
# ----------------------------------------------------------------------------
def the_data():
    import pyearthtools.tutorial

    the_path_to_the_data = get_data_minime()
    _vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "2m_temperature",
    ]
    _accessor = pyearthtools.tutorial.ERA5DataClass.ERA5LowResDemoIndex(
        _vars,
        filename_override=the_path_to_the_data,
    )
    return _accessor

def the_pipe():
    import xarray as xr
    import pyearthtools.pipeline
    import pyearthtools.data

    accessor = the_data()
    data_pipeline = pyearthtools.pipeline.Pipeline(
        accessor,
        pyearthtools.data.transforms.coordinates.StandardLongitude(
            type="-180-180",
        ),
        # Uncomment the line below if working with multi-level data
        # pyearthtools.pipeline.operations.xarray.reshape.CoordinateFlatten("level"),
        pyearthtools.pipeline.modifications.TemporalRetrieval(
            # Input = 1 sample from time T=0 hours. Output = T+6,+12,+18,+24
            concat=True,
            samples=((0, 1), (6, 4, 6)),
        ),
        # Incremental normalisation calculator
        pyearthtools.pipeline.operations.xarray.normalisation.MagicNorm(
            cache_dir=get_workdir(),
        ),
        pyearthtools.pipeline.operations.xarray.conversion.ToNumpy(),
        # channel time height width -> time channel height width

        pyearthtools.pipeline.operations.numpy.reshape.Rearrange(
            "c t h w -> t c h w",
        ), 
        sampler=pyearthtools.pipeline.samplers.Default(),
        iterator=pyearthtools.pipeline.iterators.DateRange(
            1980,
            2016,
            interval="6 hours",
        )
    )
    return data_pipeline



def the_train(max_epochs):
    import pyearthtools.training


    trainer_configuration = {
      "precision": "32",
      "checkpointing": [
        {
          "monitor": "train_loss",
          "mode": "min",
          "dirpath": "{path}/Checkpoints/Train",
          "filename": "model-{epoch:02d}-{step:02d}",
          "every_n_train_steps": 1000,
          "save_top_k": 10
        },
        {
          "monitor": "valid_loss",
          "mode": "min",
          "dirpath": "{path}/Checkpoints/Valid",
          "filename": "model-{epoch:02d}-{step:02d}-{valid_loss}",
          "every_n_train_steps": 5000,
          "save_top_k": 10
        },
        {
          "monitor": "epoch",
          "mode": "max",
          "dirpath": "{path}/Checkpoints/Epoch",
          "filename": "model-{epoch:02d}",
          "save_on_train_epoch_end": True,
          "save_top_k": 50
        }
      ]
    }

    # ---
    # Distributed data parallel 
    # - apparently lightning automagically sets everything up even if we don't
    #   explicitly use Parallel Samplers
    #
    # Note: sure if defining forkserver here is necessary, but doing it anyway.
    # from lightning.pytorch.strategies import DDPStrategy
    #
    # _strategy = DDPStrategy(accelerator="gpu", start_method="forkserver")
    _strategy = "ddp_fork"
    # ---

    # Data module
    data_pipeline = the_pipe()
    splits = {
        "train_split": pyearthtools.pipeline.iterators.DateRange(
            1980, 2016, interval="6 hours"
        ),
        "valid_split": pyearthtools.pipeline.iterators.DateRange(
            2018, 2020, interval = "6 hours"),
    }
    datamodule = pyearthtools.training.data.lightning.PipelineLightningDataModule(
        data_pipeline,
        **splits,
        **{'num_workers': 4, 'batch_size': 64}
    )

    # Model module??
    model = the_model()

    checkpoint_path = os.path.join(os.environ["ERA5LOWRESDEMO"], "chkpt")
    os.makedirs(checkpoint_path, exist_ok=True)

    _trainer = pyearthtools.training.lightning.Train(
        model.lightning_model,
        datamodule,
        path=checkpoint_path,
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer
        trainer_kwargs={
            "strategy": _strategy,
            "num_sanity_val_steps": 1,
            "max_epochs": max_epochs,
        },
        **trainer_configuration,
    )
    return _trainer

def the_model(ckpt_path=None):
    import fourcastnext
    import functools

    _pipeline = the_pipe()

    _rm_base = fourcastnext.registered_model.FourCastNextRM
    _RM = None

    if ckpt_path is None:
        _RM = _rm_base
    else:
        _RM = functools.partial(_rm_base, ckpt_path=ckpt_path)
        
    _model = _RM(
        pipeline=_pipeline,
        ckpt_path=ckpt_path,
        lightning_model_params = {
          "img_size": (64, 32), # Increase this if using additional data
          "in_channels": 4,
          "out_channels": 4, # Increase this if using additional data
          "embed_dim": 768,
          "num_blocks": 4,
          "patch_size": (2,2), # Change this to (4,4) if the GPU memory is exceeded
          "depth": 12,
        },
        output=".",
        lead_time=6  # Time delta for autoregressive step
    )

    return _model

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
def do_train():
    MAX_EPOCHS=2
    _train = the_train(MAX_EPOCHS)
    return _train.fit()

def do_predict():
    COMPARISON_BASE_TIME = '2011-03-01T00'
    COMPARISON_ANALYSIS_TIME = '2011-03-01T06'
    full_checkpoint_path = get_workdir() + '/chkpt/Checkpoints/Epoch/model-epoch=00.ckpt'
    pmodel = the_model(full_checkpoint_path)
    prediction = pmodel.run(COMPARISON_BASE_TIME)
    accessor = the_data()

    # PREDICT
    prediction = pmodel.run(COMPARISON_BASE_TIME)

    # ANALYSE
    analysis_pipeline = pyearthtools.pipeline.Pipeline(
        accessor,
        pyearthtools.data.transforms.coordinates.StandardLongitude(
            type="-180-180",
        ),
        # fourcastnext.CropToRectangleSmall(),    
        pyearthtools.pipeline.modifications.TemporalRetrieval(
            concat=True,
            samples=((0, 4, 6)),
        ),
    )

    # PLOT MODEL PREDICTIONS
    # These predictions were from training step 17,000. 
    fcst_as_celcius = prediction['2m_temperature'] - 273.15
    fcst_as_celcius.attrs["units"] = "deg C"
    fcst_as_celcius.plot.savefig(x='longitude', y='latitude', col='time', col_wrap=4)
    plt_truth.savefig("fc.png")

    # PLOT ANALYSIS GRIDS (TRUTH)
    an_as_celcius = analysis_pipeline[COMPARISON_ANALYSIS_TIME]['2m_temperature'] - 273
    an_as_celcius.attrs["units"] = "deg C"
    plt_truth = an_as_celcius.plot(x='longitude', y='latitude', col='time', col_wrap=4)
    plt_truth.savefig("an.png")

if __name__ == "__main__":
    # NAME GUARD ALL THE THINGS!
    # I'm so scared I'm even putting os in here
    import os

    mp.set_start_method("forkserver")  # ONLY FOR LINUX

    # lock CPU threads so they don"t get in the way of IO, most compute should be on the GPU
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # where the thingo stuff get saved
    if os.environ.get("ERA5LOWRESDEMO", None) is None:
        os.environ["ERA5LOWRESDEMO"] = os.path.join(os.environ["PBS_JOBFS"], "era5lowres")
    # make the thingo directory
    os.makedirs(os.environ["ERA5LOWRESDEMO"], exist_ok=True)
    EXPERIMENT_VERSION="vHACK" # not actually used so I'm gonna call it whatever I want

    # I don't even know if setting the above env variables here does anything
    # this late in a main guard, but maybe it does if processes get forked...
    # In any case its set in the runscript, its called run

    # wipe cache before starting anything
    print("--- CLEAN CACHE ---")
    print(" " * 60)
    clear_gpu_cache_and_stale_things()
    print("... done")
    print(" " * 60)

    # download data if needed
    print("--- ATTEMPT DOWNLOAD ---")
    print(" " * 60)
    download_mini_dataset()
    print("... done")
    print(" " * 60)

    train = True
    predict = True

    if train:
        print("--- TRAINING ---")
        do_train()
        print(" " * 60)
        print("... done")
        print(" " * 60)

    if predict:
        print("--- PREDICTING ---")
        do_predict()
        print(" " * 60)
        print("... done")
        print(" " * 60)

    print("--- CLEAN CACHE (again) ---")
    clear_gpu_cache_and_stale_things()
    print(" " * 60)
    print("... done")
    print(" " * 60)
