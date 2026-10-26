"""Copy the tested private-service reference layout without overwriting user code.

Objective:
- Give installed RKO users a small maintained Docker/ASGI starting point, not a template engine.

Technical approach:
- Copy an explicit canonical resource list into a newly created directory only.
- Optionally copy one reviewed same-version SDK wheel for a pre-publication/local image build.

Flow (pseudocode):
1. Reject existing destinations and invalid/mismatched wheel metadata before creating files.
2. Create a new directory exclusively and copy canonical files using exclusive writes.
3. Copy the optional SDK wheel into a dedicated wheel directory.
4. Print local build instructions without creating credentials or starting containers.

Important criteria:
- No overwrite, project introspection, implicit downloads or PyPI publication.
- The Docker build uses an explicitly selected local wheel, not an unmerged future index release.

Inputs/Outputs:
- Input: new destination and optional trusted local same-version RKO wheel.
- Output: canonical service layout ready for adapter customization and bounded Docker build.
"""

import argparse
import shutil
from collections.abc import Sequence
from email.parser import BytesParser
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from zipfile import BadZipFile
from zipfile import ZipFile


class ServiceScaffold:
    """Copy a fixed tested service layout only into an absent destination.

    Responsibilities:
    - Preserve all preexisting user files and avoid a generic template framework.
    - Support local reviewed SDK wheels without requiring release publication.

    Attributes:
        filenames (tuple[str, ...]): Explicit canonical files copied without interpolation.
    """

    filenames = (
        "app.py",
        "adapter.py",
        "Dockerfile",
        "compose.yaml",
        ".dockerignore",
        "requirements.lock",
        "README.md",
    )

    @staticmethod
    def create(destination: Path, wheel: Path | None = None) -> Path:
        """Copy reference files into a new directory, optionally with one trusted SDK wheel.

        Args:
            destination (Path): Absent target with an existing parent directory.
            wheel (Path | None): Optional reviewed same-version RKO wheel, never downloaded.

        Returns:
            Path: Created project directory.

        Raises:
            FileExistsError: If the destination exists, even when empty.
            ValueError: If the wheel is invalid, oversized or from another SDK version.
            OSError: If canonical files cannot be read/copied; partial new output may remain.
        """
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "Scaffold destination must not already exist; no files were overwritten."
            )
        if wheel is not None:
            if (
                wheel.is_symlink()
                or not wheel.is_file()
                or wheel.stat().st_size > 67_108_864
                or not wheel.name.startswith("rko_solver-")
                or not wheel.name.endswith(".whl")
                or len(wheel.name.split("-")) < 5
            ):
                raise ValueError("Provide one reviewed RKO wheel with its original valid filename.")
            try:
                with ZipFile(wheel) as archive:
                    entries = archive.infolist()
                    metadata = [
                        item for item in entries if item.filename.endswith(".dist-info/METADATA")
                    ]
                    if len(entries) > 5000 or len(metadata) != 1 or metadata[0].file_size > 65_536:
                        raise ValueError("Invalid SDK wheel metadata envelope.")
                    headers = BytesParser().parsebytes(archive.read(metadata[0]), headersonly=True)
                    if headers["Name"] != "rko-solver" or headers["Version"] != version(
                        "rko-solver"
                    ):
                        raise ValueError("Use a wheel matching this installed RKO SDK version.")
            except BadZipFile as exc:
                raise ValueError("Invalid SDK wheel archive.") from exc
        source = files("rko_solver.service").joinpath("templates")
        destination.mkdir(mode=0o700)
        for name in ServiceScaffold.filenames:
            with (destination / name).open("xb") as output:
                output.write(source.joinpath(name).read_bytes())
        wheels = destination / "wheels"
        wheels.mkdir()
        if wheel is not None:
            with wheel.open("rb") as incoming, (wheels / wheel.name).open("xb") as output:
                shutil.copyfileobj(incoming, output)
        return destination

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> None:
        """Expose a narrow scaffold command without importing optional HTTP dependencies.

        Args:
            argv (Sequence[str] | None): CLI arguments; None uses process arguments.

        Raises:
            SystemExit: For help, invalid arguments or refused/failed scaffolding.
        """
        parser = argparse.ArgumentParser(
            description="Create the bounded private RKO service reference."
        )
        commands = parser.add_subparsers(dest="command", required=True)
        scaffold = commands.add_parser(
            "scaffold", help="Copy canonical files; never overwrite a directory."
        )
        scaffold.add_argument("destination", type=Path)
        scaffold.add_argument(
            "--wheel", type=Path, help="Reviewed same-version SDK wheel for the Docker build."
        )
        args = parser.parse_args(argv)
        try:
            created = ServiceScaffold.create(args.destination, args.wheel)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Created {created}. Read README.md; no credentials or containers were created.")
        if args.wheel is not None:
            print(f"Set RKO_WHEEL={args.wheel.name} before docker compose build.")
