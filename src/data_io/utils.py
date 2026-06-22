from datetime import datetime
import os

import ROOT as r


DEFAULT_LIBRARIES = (
    "libpi_utils.so",
    "libpi_headers.so",
    "libpi_MonteCarlo.so",
    "libpi_Reco.so",
    "libPiGaudiData.so",
)


def load_pioneer_libraries(lib_dir: str = "/simulation/docker/install/lib") -> None:
    """Load the PIONEER dictionaries needed before reading ROOT objects."""
    for library in DEFAULT_LIBRARIES:
        r.gSystem.Load(f"{lib_dir}/{library}")


def print_file_creation_time(file_name: str) -> None:
    if os.path.exists(file_name):
        creation_time = os.path.getctime(file_name)
        formatted_time = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"The file '{file_name}' was created on: {formatted_time}")
    else:
        print(f"File '{file_name}' does not exist.")
