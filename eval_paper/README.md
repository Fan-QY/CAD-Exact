# Paper evaluation pipeline

These four scripts produced the submitted table and are copied here verbatim so the
rebuttal numbers land in the same metric definitions. `../eval.py` is a separate,
later implementation: it defines PEM as the fraction of positionally matching
parameters rather than the paper's sorted-integer-multiset match, samples 65536
surface points where the paper used 8192, and does not compute TVF, MFM, MFS, VGF
or TNF at all. Numbers from the two are therefore not comparable, and only these
belong in the main table.

    compute_extrude_metrics.py    -> metrics_<op>_<method>.csv   HD / CD / F@2% / vol_err / PEM / tp_tau
    compute_modeling_features.py  -> mf_<op>_<method>.csv        MFM *_count_match columns
    run_iou_5x4.py                -> iou_<op>_<method>.json      IoU
    gen_4_op_tables.py            -> LaTeX                       TVF / MFS / MFM / PEM / CD / F / HD / IoU / VGF / Validity

They address samples by index 0..1499, while the GeomCAD test splits are keyed by
the corpus ids. `results_rebuttal/index_to_geomcad_id.json` records the mapping and
`testsets_indexed/` + `predictions_indexed/` are index-named symlink views over the
real files. The links must be RELATIVE: the container mounts the workspace at /ws,
so absolute host paths dangle inside it and every metric silently reports zero
valid predictions.
