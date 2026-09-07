"""Shipped worker scripts. Each file is a complete worker (``fit``, optionally ``load``/``info``)
runnable by ``srbf/worker/runner.py`` in any interpreter that has the method's own dependencies;
none of them imports srbf. Referenced from a config by name (``worker: pysr``) or by path.
"""
