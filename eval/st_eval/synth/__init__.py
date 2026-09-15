"""Deterministic synthetic benchmark for hard Iranian e-commerce product categories.

Why synthetic: we have no licensed, labelled photos yet (HUMAN_NEEDED H-005,
docs/dataset-gap.md). Procedural rendering gives *exact* ground-truth alpha and silhouette,
and deliberately reproduces each category's known failure modes:

    carpet            fringe threads and tassels, sub-pixel gaps, beige-on-beige
    jewelry           thin gold chains, link holes, specular glints, mirror reflections
    saffron_packaged  loose saffron threads, zigzag pouch seals, clear windows
    glassware         partial transparency, refraction, clear glass on white
    handicraft        scalloped rims, handle holes, micro-mosaic texture vs patterned backgrounds
    clothing_hanger   wire hooks, sleeve gaps, sheer scarves with fringe, fuzzy knit edges

Ground-truth convention (documented in docs/dataset-gap.md):
    mask/  = the product silhouette, i.e. what a seller expects in the cutout (glass included in full)
    alpha/ = physical opacity (glass is partly transparent)
Shadows and reflections on the surface are background.
"""

from .generate import CATEGORY_COUNTS, GENERATOR_VERSION, generate_dataset, generate_sample

__all__ = ["CATEGORY_COUNTS", "GENERATOR_VERSION", "generate_dataset", "generate_sample"]
