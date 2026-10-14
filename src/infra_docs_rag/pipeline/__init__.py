"""The pipeline as separable stages: what each one reads and writes, the version of everything that
produced an artifact, and the controls a demo leaves out: skipping what is up to date, stopping at the
stage that failed, retrying it alone, building an index next to the live one before it goes live, and
rolling a release back."""
