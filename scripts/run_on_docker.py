import argparse
import pathlib
import shutil

import docker
from loguru import logger

cwd_path = pathlib.Path.cwd()
# Setup argument parser
parser = argparse.ArgumentParser(
    description="Copy files to/from a Docker container and execute python script inside it."
)
parser.add_argument(
    "--input_code_file", required=False, default="scripts/create_dataset.py"
)
parser.add_argument(
    "--src_dir",
    required=False,
    help="Path to the src directory",
    default=cwd_path.joinpath("."),
)
parser.add_argument(
    "--out_dir",
    required=False,
    help="Path to the host output directory",
    default=cwd_path,
)
parser.add_argument(
    "--container_name",
    required=False,
    help="Name of the FeniCS docker container",
    default="dolfinx",
)
parser.add_argument(
    "--working_dir_container",
    required=False,
    help="Base directory in container",
    default="/home/",
)
parser.add_argument(
    "--is_complex",
    action="store_true",
    required=False,
    help="Toggle for complex valued execution",
    default=True,
)

# Parse arguments
args = parser.parse_args()

# Assign variables from arguments
#   host
project_file = cwd_path.joinpath("pyproject.toml")
py_file = cwd_path.joinpath(args.input_code_file)
readme_file = cwd_path.joinpath("README.md")
license_file = cwd_path.joinpath("LICENSE")
host_src_dir = cwd_path.joinpath(args.src_dir)
host_output_dir = cwd_path.joinpath(args.out_dir)
#   target
container_name = args.container_name
container_working_dir = pathlib.Path(args.working_dir_container)
container_src_dir = "hps"
is_complex = args.is_complex

# Complex build toggle
builds = {
    True: "/usr/local/bin/dolfinx-complex-mode",
    False: "/usr/local/bin/dolfinx-real-mode",
}

client = docker.from_env()


def copy_to_container(
    container, host_entry: pathlib.Path, container_dir: pathlib.Path
) -> None:
    """Copies host dir to container dir

    Args:
        container:
        host_entry:
        container_dir:
    """
    if host_entry.is_dir():
        tar_stream = shutil.make_archive("tmp", "tar", host_entry)
        logger.info(
            f"Copying {host_entry} to {container_dir} ({len(list(host_entry.glob('*')))} files)."
        )
    elif host_entry.is_file():
        tar_stream = shutil.make_archive(
            "tmp", "tar", host_entry.parent, host_entry.name
        )
        logger.info(f"Copying {host_entry} to {container_dir}.")
    else:
        raise ValueError(f"Unknown host file system entry: {host_entry}!")
    with open(tar_stream, "rb") as file_obj:
        container.put_archive(container_dir, file_obj)
    cwd_path.joinpath("tmp.tar").unlink()


def execute_script_in_container(container, script_path: pathlib.Path) -> None:
    """Executes a script in container

    Args:
        container:
        script_path:
    """
    _, stream = container.exec_run(
        f"bash -c 'source {builds[is_complex]}; "
        f"pip install .;"
        f"python3 {script_path.name}'",
        workdir=str(container_working_dir),
        stream=True,
    )

    for data in stream:
        print(data.decode(), end="")


def cleanup_container(container, clean_dir) -> None:
    """Deletes files in container base dir.

    Args:
        container:
        clean_dir:
    """
    ec, _ = container.exec_run(f"rm -rf {clean_dir}/*")
    if ec != 0:
        raise Exception(f"Directory cleanup failed with exit code {ec}!")


def setup_container(container, working_dir: pathlib.Path) -> None:
    """Makes out and src dir.

    Args:
        container:
        working_dir:
    """
    container.exec_run(f"mkdir -p {working_dir.joinpath(container_src_dir)}")
    container.exec_run("apt-get install python3-pip")


def copy_from_container(
    container, container_dir: pathlib.Path, host_dir: pathlib.Path
) -> None:
    """Copies output files from container to output directory on host.

    Args:
        container:
        container_dir:
        host_dir:
    """
    stream, stat = container.get_archive(str(container_dir))
    with open("output.tar", "wb") as out_file:
        for chunk in stream:
            out_file.write(chunk)
    shutil.unpack_archive("output.tar", host_dir)
    cwd_path.joinpath("output.tar").unlink()


if __name__ == "__main__":
    try:
        logger.info(f"Starting with {args}.")
        # Create or get container
        container_ = client.containers.get(container_name)
        container_.start()
        logger.info(f"Started docker container {container_name}.")

        # setup container
        setup_container(container_, container_working_dir)
        logger.info("Successfully setup docker container.")

        # Copy relevant files to container
        copy_to_container(
            container_, host_src_dir, container_working_dir.joinpath(container_src_dir)
        )
        copy_to_container(container_, project_file, container_working_dir)
        copy_to_container(container_, readme_file, container_working_dir)
        copy_to_container(container_, license_file, container_working_dir)
        copy_to_container(container_, py_file, container_working_dir)
        logger.info("Copied all files to container.")

        logger.info("Starting simulation....")
        # Set environment variables and execute the script
        execute_script_in_container(container_, py_file)
        logger.info("Finished simulation....")

        # ensure proper output dir exists for host
        host_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy output files to host
        copy_from_container(
            container_,
            container_working_dir.joinpath("data"),
            host_output_dir,
        )
        logger.info(f"Finished copying files from container to {host_output_dir}.")

    finally:
        client.close()
