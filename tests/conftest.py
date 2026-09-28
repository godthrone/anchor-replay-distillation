# conftest.py — Shared pytest fixtures and configuration.
# Responsibility: provide reusable test fixtures (ontology, configs, sample
# data) for all test modules in the test suite.

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from ard.backends.ontology_loader import load_ontology_v4
from ard.core.ontology import OntologyV4
from ard.core.types import AnchorSpec, TurnSpec

#: The ontology the production pipeline reads (see ``configs/config.toml``).
ONTOLOGY_PATH = "ontology/anchor_ontology.v4.json"


@pytest.fixture(scope="module")
def ontology() -> OntologyV4:
    """Load the shared v4 ontology for tests."""
    return load_ontology_v4(ONTOLOGY_PATH)


def _one_pixel_png(shade: int) -> bytes:
    """A real, minimal, decodable 1x1 RGB PNG whose bytes depend on *shade*."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes([0, shade, shade, shade]))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@dataclass(frozen=True)
class DomainImages:
    """One byte-distinct real picture per ``visual_domain``, production-resolved.

    The guards that probe the image branch need a picture that is *actually*
    different per domain — otherwise "the address changed" would not mean "the
    bytes the model receives changed", and a guard could pass on a coordinate
    whose picture never reached the request.  Resolution goes through the
    **production** entry point (:func:`ard.domain.image_store.resolve_domain_images`),
    not a hand-rolled path join, so the addressing rule under test is the real one.
    """

    root: Path
    selected: Mapping[str, Path]
    digests: Mapping[str, str]

    def picture(self, visual_domain: str | None) -> Path | None:
        """The resolved picture for *visual_domain* (``None`` for text-only)."""
        if visual_domain is None:
            return None
        return self.selected.get(visual_domain)

    def data_url(self, visual_domain: str | None) -> str | None:
        """The inline base64 data URI of that picture (``None`` when there is none)."""
        path = self.picture(visual_domain)
        if path is None:
            return None
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def address(self, visual_domain: str | None) -> str:
        """The picture's tree address, as the render's ``V`` channel states it."""
        path = self.picture(visual_domain)
        return "" if path is None else f"<image_dir>/{path.parent.name}/{path.name}"


@pytest.fixture(scope="module")
def domain_images(ontology: OntologyV4, tmp_path_factory: pytest.TempPathFactory) -> DomainImages:
    """Build the per-``visual_domain`` picture tree and resolve it in production code.

    Distinct bytes per domain are the point: a coordinate's ``visual_domain`` is
    the only thing selecting its picture, so two domains sharing a picture would
    silently weaken every "the picture the coordinate carries" assertion.
    """
    from ard.domain.image_store import list_domain_images, resolve_domain_images

    root = tmp_path_factory.mktemp("domain-images")
    domains = list(ontology.axis_values("visual_domain"))
    for index, domain in enumerate(domains):
        directory = root / domain
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pic.png").write_bytes(_one_pixel_png(shade=(index * 11 + 5) % 256))
    for domain in domains:
        assert list_domain_images(root, domain), f"{domain}: fixture picture not found"

    specs = [
        AnchorSpec(
            id=f"fixture-{domain}",
            anchor_meta={"visual_domain": domain, "modality": "image"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for domain in domains
    ]
    resolution = resolve_domain_images(root, specs, seed=1488, cycle=0)
    assert not resolution.missing, f"fixture did not resolve: {sorted(resolution.missing)}"

    selected = dict(resolution.selected)
    digests = {
        domain: hashlib.sha256(selected[domain].read_bytes()).hexdigest()[:16] for domain in domains
    }
    assert len(set(digests.values())) == len(domains), "fixture pictures are not byte-distinct"
    return DomainImages(root=root, selected=selected, digests=digests)
