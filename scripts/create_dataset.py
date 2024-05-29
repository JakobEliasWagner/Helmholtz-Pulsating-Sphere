import pathlib

import numpy as np

import hps


def main():
    # run properties
    msh = hps.MeshProperties(elements_per_wavelengths=10)
    dom = hps.DomainProperties()
    phy = hps.PhysicsProperties()

    # boundaries
    # top boundary
    top_sampler = hps.MeshGrid([0.0, -1.0], [1.0, 0.0])
    top_func = hps.Function(
        lambda a: lambda x: np.ones(x[0].shape) * a[0] + 1j * a[1] * np.ones(x[0].shape)
    )

    # right boundary
    right_sampler = hps.MeshGrid([0.0, -1.0], [1.0, 0.0])
    right_func = hps.Function(
        lambda a: lambda x: np.ones(x[1].shape) * a[0] + 1j * a[1] * np.ones(x[1].shape)
    )

    """top_sampler = hps.MeshGrid([1.], [0.])
    top_func = hps.Function(
        lambda a: lambda x: np.ones(x[0].shape) * 1e-3 + 1j * a[0] * np.ones(x[0].shape)
    )

    # right boundary
    right_sampler = hps.MeshGrid([1.], [0.])
    right_func = hps.Function(
        lambda a: lambda x: np.ones(x[1].shape) * 1e-3 + 1j * a[0] * np.ones(x[1].shape)
    )"""

    # frequencies
    frequency_sampler = hps.UniformSampler([500.0], [500.0])

    # assemble run description
    n_observations = 2**12
    run = hps.RunProperties(
        domain=dom,
        mesh=msh,
        physics=phy,
        frequency_sampler=frequency_sampler,
        top_sampler=top_sampler,
        top_boundary=top_func,
        right_sampler=right_sampler,
        right_boundary=right_func,
        n_observations=n_observations,
    )

    # setup helmholtz configuration
    h = hps.Helmholtz(run)

    # out dir
    run_id = hps.UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("data", str(run_id))
    out_dir.mkdir(parents=True, exist_ok=False)

    # run
    h(out_dir)


if __name__ == "__main__":
    main()
