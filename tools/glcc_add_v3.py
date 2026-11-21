"""Add an experimental 'v3' target to the installed glcc (ROMv3 = v4 vCPU code, romType 0x28).

ROMv3 and ROMv4 share the vCPU instruction set and the SYS functions TTLM uses;
glcc simply refuses to target v3. Usage: python tools/glcc_add_v3.py
"""
import pathlib
p = pathlib.Path(__file__).resolve().parent.parent / "third_party/glcc-install/lib/gigatron-lcc/roms.json"
s = p.read_text()
if '"v3"' not in s:
    s = s.replace('    "v4"    : { "cpu" : 4,', '    "v3"    : { "cpu" : 4, "inherits" : "v4", "romType" : "0x28",\n'
                  '\t\t"info" : "TTLM: ROMv3 with v4 code (experimental)" },\n    "v4"    : { "cpu" : 4,', 1)
    p.write_text(s)
print("glcc target v3:", '"v3"' in p.read_text())
