import pathlib
import time

import numpy as np

import hps


def main():
    # run properties
    msh = hps.MeshProperties()
    dom = hps.DomainProperties()
    phy = hps.PhysicsProperties()

    # boundaries
    n_samples = 2**9
    # top boundary
    top_parameters = np.random.rand(n_samples, 2)
    top_func = hps.Function(lambda a: lambda x: x[0] * 0 + a[0] - 1j * a[1])

    # right boundary
    right_parameters = np.random.rand(n_samples, 2)
    right_func = hps.Function(lambda a: lambda x: x[0] * 0 + a[0] - 1j * a[1])

    # assemble run description
    run = hps.RunProperties(
        domain=dom,
        mesh=msh,
        physics=phy,
        frequencies=np.linspace(100, 500, 257),
        top_boundary=top_func,
        top_parameters=top_parameters,
        right_boundary=right_func,
        right_parameters=right_parameters,
    )

    # setup helmholtz configuration
    h = hps.Helmholtz(run)

    # out dir
    run_id = hps.UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("data", str(run_id))
    out_dir.mkdir(parents=True, exist_ok=False)

    # run
    start = time.time()
    h(out_dir)
    end = time.time()
    print(f"Execution took {end - start}s")


if __name__ == "__main__":
    main()
