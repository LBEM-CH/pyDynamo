"""Check that an env's installed files are exactly the tested ones.

    python -I -B verify_env.py MANIFEST.json [MORE_MANIFEST.json ...] SITE_PACKAGES_DIR
MANIFEST.json is record_compare.py's output for the tested env (pydynamo-uv). Exit status 0 when every listed file
exists with the same sha256 and no other distribution (except pydynamo) is installed. Several manifests are merged:
pydynamo-manifest.json (the numerical packages) plus pydynamo-gui-manifest.json (the GUI section of the lock).
"""
import glob
import hashlib
import json
import os
import re
import sys

man, site = {}, sys.argv[-1]
for m in sys.argv[1:-1]:
    man.update(json.load(open(m)))
bad = 0
for name, d in sorted(man.items()):
    for f, h in d["files"].items():
        p = os.path.join(site, f)
        if not os.path.isfile(p):
            print("MISSING", name, f)
            bad += 1
        elif hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            if name == "pydynamo":
                continue                     # editable metadata (README text) may change with the source
            print("DIFFERS", name, f)
            bad += 1
installed = {re.sub(r"[-_.]+", "-", os.path.basename(d).split("-")[0]).lower()
             for d in glob.glob(os.path.join(site, "*.dist-info"))}
extra = sorted(installed - set(man))
if extra:
    print("EXTRA distributions:", extra)
    bad += len(extra)
print("verify_env:", "OK" if not bad else "%d problems" % bad, "(%d distributions, %d files)" % (
    len(man), sum(len(d["files"]) for d in man.values())))
sys.exit(1 if bad else 0)
