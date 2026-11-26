/* When each published method was first made public: its earliest preprint, paper, public code, package or model,
 * whichever came first, with what that was and a link to it. The explorer's "Over the years" view places every method
 * at this date. Kept by hand, one entry per method key of the release; the release refresh writes only
 * data/<release>/, so it never touches this file. A method without an entry has no date and is not drawn there (the
 * Oracle is a reference, not a published method). date: YYYY-MM-DD, or YYYY-MM where only the month is known. */
window.PUBLICATION_DATES = {
  "gpgomea": { "date": "2017-07", "precision": "month", "event": "conference paper at GECCO 2017", "url": "https://doi.org/10.1145/3071178.3071287" },
  "operon": { "date": "2019-09-10", "precision": "day", "event": "code on GitHub", "url": "https://github.com/heal-research/operon" },
  "dsr": { "date": "2019-09-25", "precision": "day", "event": "submission to ICLR 2020, public on OpenReview", "url": "https://openreview.net/forum?id=S1gKA6NtPS" },
  "qlattice": { "date": "2020-03-04", "precision": "day", "event": "first release of its Python package, feyn, on PyPI", "url": "https://pypi.org/project/feyn/1.0.1/" },
  "PySR": { "date": "2020-09-14", "precision": "day", "event": "code on GitHub", "url": "https://github.com/MilesCranmer/PySR/commit/688106d28004a4c969c89d5dd8316f04144c7972" },
  "nesymres-100M": { "date": "2021-06-11", "precision": "day", "event": "preprint on arXiv", "url": "https://arxiv.org/abs/2106.06427v1" },
  "e2e": { "date": "2022-04-22", "precision": "day", "event": "preprint on arXiv", "url": "https://arxiv.org/abs/2204.10532v1" },
  "udsr": { "date": "2022-05-11", "precision": "day", "event": "entry to the SRBench 2022 competition, with its code, on GitHub", "url": "https://github.com/cavalab/srbench/pull/119" },
  "rilsrols": { "date": "2022-11-27", "precision": "day", "event": "first release on PyPI", "url": "https://pypi.org/project/rils-rols/0.1/" },
  "T8-3M": { "date": "2026-09-14", "precision": "day", "event": "model on Hugging Face, with flash-ansr 0.17.0", "url": "https://github.com/psaegert/flash-ansr/releases/tag/v0.17.0" },
  "T8-20M": { "date": "2026-09-14", "precision": "day", "event": "model on Hugging Face, with flash-ansr 0.17.0", "url": "https://github.com/psaegert/flash-ansr/releases/tag/v0.17.0" },
  "T8-120M": { "date": "2026-09-14", "precision": "day", "event": "model on Hugging Face, with flash-ansr 0.17.0", "url": "https://github.com/psaegert/flash-ansr/releases/tag/v0.17.0" },
  "T8-120M-pysr": { "date": "2026-09-29", "precision": "day", "event": "first results, on this site", "url": "https://github.com/psaegert/srbf/pull/53" }
};
