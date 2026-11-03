"""One model entry point for training, checkpoint loading and comparisons."""


def build_model(config):
    if config.model.backbone == "functional_tucker":
        from .functional import NornFunctionalTucker

        return NornFunctionalTucker(
            config.grid, config.model, config.max_age_ma, config.anchor_step_myr
        )
    from .operator import NornSFNO

    return NornSFNO(config.grid, config.model, config.max_age_ma)
