# Figure QA notes

The September 2026 figure revision changed explanatory labels and operating-point annotations only. No experiment rows, benchmark values, or masks were altered.

- `benchmark_summary`, `advantage_heatmap`, `selection_transition_heatmap`, and `risk_coverage_all_generators` passed the render-time panel-alignment gate. The corresponding JSON reports are stored in this directory.
- PDF text audits found no glyph below 5 pt in the four Matplotlib figures.
- The collision audit passed without findings for `advantage_heatmap`, `selection_transition_heatmap`, and `risk_coverage_all_generators`.
- The benchmark collision report flags hatch strokes and broad title/legend bounding boxes. Final-size visual inspection confirms that the labels, legend, panel titles, and bars do not intersect after the legend was moved to a single row.
- The two framework PDFs are exported from editable SVG with CairoSVG. CairoSVG emits 1 pt text runs under scaling transforms, so the PDF text scanner cannot infer their effective sizes. The authoritative SVG font sizes are 5.6--7.8 pt, and final-size visual inspection confirms legibility.
- The framework collision reports merge nearby SVG text lines into common bounding boxes and treat surrounding box outlines as text-stroke intersections. Visual inspection confirms that the text is contained within its nodes, the graphical-abstract proposal route avoids the gate title, and the detailed-framework title is not clipped.

Files ending in `.collision-audit.pdf` and `.alignment.svg` are diagnostic overlays only and are not manuscript assets.
