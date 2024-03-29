import pathlib
import time

import hps


def main():
    # run properties
    msh = hps.MeshProperties()
    dom = hps.DomainProperties()
    phy = hps.PhysicsProperties()

    # boundaries
    # top boundary
    top_sampler = hps.UniformSampler([0, -1], [1, 0])
    top_func = hps.Function(lambda a: lambda x: x[0] * 0 + a[0] - 1j * a[1])

    # right boundary
    right_sampler = hps.UniformSampler([0, -1], [1, 0])
    right_func = hps.Function(lambda a: lambda x: x[0] * 0 + a[0] - 1j * a[1])

    # frequencies
    frequency_sampler = hps.UniformSampler([100], [500])

    # assemble run description
    n_observations = 2**5
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
    start = time.time()
    h(out_dir)
    end = time.time()
    print(f"Execution took {end - start}s")


if __name__ == "__main__":
    main()
