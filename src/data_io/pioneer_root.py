import ROOT as r


DEFAULT_LIBRARIES = (
    "libpi_utils.so",
    "libpi_headers.so",
    "libpi_MonteCarlo.so",
    "libpi_Reco.so",
    "libPiGaudiData.so",
)


def load_pioneer_libraries(lib_dir: str = "/simulation/docker/install/lib") -> None:
    for library in DEFAULT_LIBRARIES:
        r.gSystem.Load(f"{lib_dir}/{library}")


class GeoHeaderHandle:
    def __init__(self, path: str):
        self.file = r.TFile(path, "READ")
        self.header = self.file.Get("GeoHeader")
        if not self.header:
            raise RuntimeError(f"No GeoHeader found in {path}")

    def __getattr__(self, name: str):
        return getattr(self.header, name)


def open_geo_header(path: str) -> GeoHeaderHandle:
    return GeoHeaderHandle(path)
