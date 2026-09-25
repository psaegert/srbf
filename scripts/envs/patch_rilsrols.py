#!/usr/bin/env python3
"""Make rils-rols 1.6.7 print the constants of its final model with every digit.

    python scripts/envs/patch_rilsrols.py SOURCE_DIR      # the unpacked rils-rols-1.6.7 sdist

rils-rols prints a model with ``node::to_string``, which writes a constant with ``std::to_string``: six decimals,
so that 6.674e-11 becomes ``0.000000``. The search itself also uses ``to_string``, as the key that deduplicates
candidates and skips perturbations it has already checked, so changing what ``to_string`` prints would change the
search. The patch therefore adds a flag that only ``get_model_string`` (the final model handed to Python) sets:
while it is set, a constant is printed with ``%.17g``, which reads back as the same double. The method's own rule
that prints a constant within 1e-12 of an integer as that integer is kept where the integer is not 0 (a change of at
most 1e-12, printed at full precision: ``std::to_string`` of the ``int`` cast overflowed beyond 2^31). A nonzero
constant within 1e-12 of 0 is printed as it is: the rule printed it as 0, which turned a division by it into a
division by zero. Nothing else changes. Used by ``scripts/envs/build_rilsrols_env.sh``.
"""
from __future__ import annotations

import sys
from pathlib import Path

FLAG = "full_precision_constants"

# (file, text to find, text to put in its place); every text must occur exactly once. Lines are written with "\n"
# and matched in the file's own line endings (the sdist's C++ files end their lines with CRLF).
EDITS: list[tuple[str, str, str]] = [
    ("rils_rols_cpp/node.h",
     "#include <memory>\n",
     "#include <memory>\n#include <cstdio>\n"),
    ("rils_rols_cpp/node.h",
     "# define EPS pow(10, -PRECISION)\n",
     "# define EPS pow(10, -PRECISION)\n"
     "\n"
     "// srbf: set only while get_model_string prints the final model, so that its constants carry every digit\n"
     "// (%.17g); the search keys candidates by to_string and is unchanged\n"
     f"inline bool {FLAG} = false;\n"),
    ("rils_rols_cpp/node.h",
     "\t\tcase node_type::CONST:\n\t\t\tif (abs(std::round(const_value) - const_value) < EPS) \n",
     "\t\tcase node_type::CONST:\n"
     f"\t\t\tif ({FLAG}) {{ const double r = std::round(const_value); char buf[32]; snprintf(buf, sizeof buf, "
     "\"%.17g\", abs(r - const_value) < EPS && r != 0 ? r : const_value); return buf; }\n"
     "\t\t\tif (abs(std::round(const_value) - const_value) < EPS) \n"),
    ("rils_rols_cpp/rils_rols_cpp.cpp",
     "\tstring get_model_string() {\n\t\treturn final_solution->to_string();\n\t}\n",
     "\tstring get_model_string() {\n"
     f"\t\t{FLAG} = true;   // srbf: see node.h\n"
     "\t\tconst auto model = final_solution->to_string();\n"
     f"\t\t{FLAG} = false;\n"
     "\t\treturn model;\n"
     "\t}\n"),
]


def patch(source: Path) -> None:
    texts: dict[str, bytes] = {}
    for name, old, new in EDITS:
        path = source / name
        text = texts.get(name)
        if text is None:
            text = path.read_bytes()
            if FLAG.encode() in text:
                raise SystemExit(f"{path} is already patched")
        newline = "\r\n" if b"\r\n" in text else "\n"
        find = old.replace("\n", newline).encode()
        count = text.count(find)
        if count != 1:
            raise SystemExit(f"{path}: expected the text to replace exactly once, found it {count} times:\n{old}")
        texts[name] = text.replace(find, new.replace("\n", newline).encode())
    for name, text in texts.items():
        (source / name).write_bytes(text)
        print(f"patched {source / name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    patch(Path(sys.argv[1]))
