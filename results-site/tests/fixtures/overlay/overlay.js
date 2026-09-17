// A stand-in overlay for the tests: one method, one complete cell, the same schema as any release payload.
// Sealed into sealed_fixture.js with the key in site_v2.spec.mjs, so the test drives the real unlock path.
window.RESULTS_V2_PRIVATE = {
  "base": "",
  "methods": [{ "key": "fixture-method", "label": "Fixture Method", "param": "draws per problem", "budget": "candidates",
    "provenance": "harness_tuned", "selection": "Fixture: the lowest score of the drawn candidates.", "color": "#7a5195" }],
  "cells": { "fixture-method": { "constant": { "1": { "state": "complete",
    "r": { "numeric_recovery_val": [3, 10], "symbolic_recovery": [1, 10], "success": [10, 10] },
    "m": { "log10_fvu_val": [10, 10, -30.0, 120.0], "mdl_ratio": [10, 10, 2.0, 1.0], "fit_time": [10, 10, 5.0, 3.0] } } } } },
  "status": { "fixture-method": [1, 1] },
  "timing": {}
};
