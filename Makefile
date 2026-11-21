# TTLM build. Needs third_party/glcc-install (gigatron-lcc) and the Python venv.
GLCC = third_party/glcc-install/bin/glcc
GTSIM = third_party/glcc-install/bin/gtsim
ROM = third_party/gigatron-lcc/gigatron/roms/v4.rom
PY = .venv/bin/python
SRC = gigatron/src

all: gigatron/ttlm.gt1 gigatron/ttlm_v3.gt1

$(SRC)/weights.c: model/moth.pt model/export_c.py
	cd model && ../$(PY) export_c.py moth.pt ../$(SRC)/weights.c

gigatron/ttlm.gt1: $(SRC)/weights.c $(SRC)/engine.c $(SRC)/row.s $(SRC)/ttlm.c
	cd $(SRC) && ../../$(GLCC) -rom=v4 -map=32k -O engine.c row.s weights.c ttlm.c -o ../build_v4.gt1 --frags > ../build_v4.frags
	$(PY) tools/gt1_fill.py gigatron/build_v4.gt1 $@

# same program for ROMv3 (experimental glcc target, see tools/glcc_add_v3.py)
gigatron/ttlm_v3.gt1: $(SRC)/weights.c $(SRC)/engine.c $(SRC)/row.s $(SRC)/ttlm.c
	$(PY) tools/glcc_add_v3.py
	cd $(SRC) && ../../$(GLCC) -rom=v3 -map=32k -O engine.c row.s weights.c ttlm.c -o ../build_v3.gt1
	$(PY) tools/gt1_fill.py gigatron/build_v3.gt1 $@

# bit-exactness: C on the Gigatron simulator vs the Python integer reference
test: $(SRC)/weights.c
	cd model && ../$(PY) make_tests.py moth.pt ../$(SRC)/tests.h > ../$(SRC)/expected.txt
	cd $(SRC) && ../../$(GLCC) -rom=v4 -map=sim -O engine.c row.s weights.c test_sim.c -o ../test_sim.gt1
	$(GTSIM) -rom=$(ROM) gigatron/test_sim.gt1 > gigatron/got.txt
	diff <(sed 's/: */:/' $(SRC)/expected.txt | tr -s ' ') <(sed 's/: */:/' gigatron/got.txt | tr -s ' ') && echo BIT-EXACT

# the browser version (GitHub Pages): web/ + the program + the address of ttlm_state
web: gigatron/ttlm.gt1
	cp gigatron/ttlm.gt1 web/ttlm.gt1
	python3 -c "import re,json,sys; sys.path.insert(0,'tools'); from brainmap import brainmap; m=re.search(r'([0-9a-f]{4})-[0-9a-f]{4} \\(1 byte\\)\\s+BSS\\s+ttlm_state\\b', open('gigatron/build_v4.frags').read()); json.dump({'state': int(m.group(1), 16), 'brain': brainmap('gigatron/build_v4.frags')}, open('web/ttlm.json', 'w'), separators=(',', ':'))"
	python3 -c "import json; d=json.load(open('web/ttlm.json')); print('state', hex(d['state']), 'brain', len(d['brain']['kind']))"

serve: web
	cd web && python3 -m http.server 8765

tools/gtshot: tools/gtshot.c
	cc -O2 -o $@ $<

.PHONY: all test web serve
SHELL = /bin/bash
