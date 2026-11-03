import pytest


@pytest.fixture
def tiny_problem(tmp_path):
    from norn_earth.config import load_config
    from norn_earth.data.demo import create_demo
    from norn_earth.data.preparation import load_dataset, prepare_dataset

    path = create_demo(
        tmp_path / "fixture",
        device="cpu",
        nlat=4,
        nlon=8,
        width=4,
        blocks=1,
        steps=3,
        anchor_step=20,
    )
    config = load_config(path).with_tag("no_physics")
    config.model.backbone = "functional_tucker"
    config.model.spatial_rank = 2
    config.model.context_rank = 2
    config.model.head_mode = "input_residual"
    config.training.selection_metric = "validation_mae"
    config.training.latent_gp_weight = 0.001
    config.training.num_threads = 1
    config.training.log_every = 2
    config.training.validation_every = 1
    config.training.checkpoint_every = 1
    config.output_dir = str(tmp_path / "run")
    dataset = prepare_dataset(config, tmp_path / "dataset.npz")
    config.data.prepared = str(dataset)
    return config, load_dataset(dataset, config), dataset
