// TTLM in the browser: a Gigatron (Philip Thomas' simulator core) that boots
// straight into the moth, with a text box that types into the machine.
import { Gigatron } from './emu/gigatron.js';
import { Vga } from './emu/vga.js';

const HZ = 6250000;
const VSYNC = 0x80;
const SYS_EXEC = 0x00ad, SYSFN = 0x22, VPC = 0x16, VLR = 0x1a;

const $ = (id) => document.getElementById(id);
const canvas = $('screen');
const status = $('status');
const speedLabel = $('speed');

const cpu = new Gigatron({ hz: HZ, romAddressWidth: 16, ramAddressWidth: 15 });
const vga = new Vga(canvas, cpu, {
  horizontal: { frontPorch: 16, pulse: 96, backPorch: 48, visible: 640 },
  vertical: { frontPorch: 6, pulse: 8, backPorch: 27, visible: 480 },
});

let gt1 = null;          // program bytes
let stateAddr = -1;      // address of ttlm_state in the Gigatron's RAM
let execCount = 0;
let turbo = true;        // faster than the real machine while the moth works
let cycles = 0;          // Gigatron clock cycles since boot
let frames = 0;
let lastOut = 0;

// ---- loading: the second SYS_Exec (the main menu) runs TTLM instead ----
function loadGt1(bytes) {
  let i = 0, addr;
  do {
    addr = (bytes[i] << 8) | bytes[i + 1];
    let n = bytes[i + 2] || 256;
    i += 3;
    while (n--) {
      cpu.ram[addr & 0x7fff] = bytes[i++];
      addr = (addr & 0xff00) | ((addr + 1) & 0xff);
    }
  } while (bytes[i] !== 0);
  return (bytes[i + 1] << 8) | bytes[i + 2];
}

function doke(a, v) { cpu.ram[a] = v & 255; cpu.ram[a + 1] = (v >> 8) & 255; }
function deek(a) { return cpu.ram[a] | (cpu.ram[a + 1] << 8); }

function step() {
  if (cpu.nextpc === 0x3b4 && deek(SYSFN) === SYS_EXEC) {
    const vpc = deek(VPC);
    if (!(execCount === 0 && vpc >= 0x1f0 && vpc < 0x1f8) && ++execCount === 1 && gt1) {
      const start = loadGt1(gt1);
      doke(VPC, (start & 0xff00) | ((start - 2) & 0xff));
      doke(VLR, start);
      cpu.ac = 0xf8;                       // the instruction replaced: LD $f8
      cpu.pc = 0x3cb;                      // REENTER
      cpu.nextpc = 0x3cc;
      vga.tick();
      return;
    }
  }
  cpu.tick();
  vga.tick();
  cycles++;
  const out = cpu.out;
  if ((lastOut & ~out) & VSYNC) { frames++; onFrame(); }
  lastOut = out;
}

// ---- typing into the machine: each key held 3 frames, released 3 frames ----
let typing = [];         // pending key codes
let keyPhase = 0;
let queued = null;       // question waiting for the moth to be ready

// time of one answer, in Gigatron time and in browser time
let lastState = 0, t0 = 0, w0 = 0, lastAnswer = null;
function onFrame() {
  const s = mothState();
  if (s !== lastState) {
    if (s === 2) { t0 = cycles; w0 = performance.now(); }
    if (s === 1 && lastState === 2) lastAnswer = { gt: (cycles - t0) / HZ, wall: (performance.now() - w0) / 1000 };
    lastState = s;
  }
  if (typing.length) {
    if (keyPhase === 0) cpu.inReg = typing[0];
    if (keyPhase === 3) cpu.inReg = 0xff;
    if (++keyPhase === 6) { typing.shift(); keyPhase = 0; }
    return;
  }
  if (queued !== null && mothState() === 1) {
    typing = [...queued].map((c) => c.charCodeAt(0)).concat([10]);
    queued = null;
    keyPhase = 0;
  }
}

function mothState() { return stateAddr >= 0 ? cpu.ram[stateAddr] : 0; }

function ask(text) {
  const q = text.toLowerCase().replace(/[^a-z0-9 ?.,!']/g, '').replace(/\s+/g, ' ').trim().slice(0, 30);
  if (!q || typing.length) return;
  queued = q;
  updateStatus();
}

// ---- run loop: real time (1x) or as fast as this computer allows ----
let simTime = 0, wallTime = 0;
let last = performance.now();
function loop(now) {
  const dt = Math.min(now - last, 100);
  last = now;
  const budget = performance.now() + 14;               // ms of CPU per animation frame
  const busy = typing.length > 0 || queued !== null || mothState() !== 1;
  const target = turbo && busy ? Infinity : Math.round(HZ * dt / 1000);
  let done = 0;
  while (done < target) {
    for (let i = 0; i < 2000; i++) step();
    done += 2000;
    if (performance.now() > budget) break;
  }
  simTime += done / HZ; wallTime += dt / 1000;
  if (xrayOn) drawXray();
  requestAnimationFrame(loop);
}

let shownSpeed = '';
function updateStatus() {
  const s = mothState();
  let text = 'Booting…';
  if (s === 1) text = queued ? 'Typing…' : lastAnswer
    ? `Your turn. That answer took ${Math.round(lastAnswer.gt)} s on a real Gigatron` +
      (turbo && lastAnswer.wall < lastAnswer.gt * 0.8 ? ` (${Math.round(lastAnswer.wall)} s here).` : '.')
    : 'The moth is listening. Ask it something.';
  if (s === 2) text = queued ? 'The moth is busy. Your question will be typed when it is done.'
                             : 'The moth is thinking…';
  if (typing.length) text = 'Typing…';
  status.textContent = text;
  if (wallTime > 0.5) {
    const x = simTime / wallTime;
    const sp = x >= 0.95 && x <= 1.05 ? 'real speed' : `${x.toFixed(1)}× real speed`;
    if (sp !== shownSpeed) { speedLabel.textContent = sp; shownSpeed = sp; }
    simTime = wallTime = 0;
  }
}
setInterval(updateStatus, 500);

// ---- UI ----
$('ask').addEventListener('submit', (e) => {
  e.preventDefault();
  ask($('q').value);
  $('q').value = '';
});
for (const b of document.querySelectorAll('.chip')) b.addEventListener('click', () => ask(b.textContent));
$('realspeed').addEventListener('change', (e) => { turbo = !e.target.checked; });
$('reset').addEventListener('click', () => {
  cpu.reset(); execCount = 0; typing = []; queued = null; lastAnswer = null; lastState = 0;
});

// ---- X-ray: what each pixel of the brain holds, drawn live from the RAM ----
const xray = $('xray');
const xctx = xray.getContext('2d');
const ximg = xctx.createImageData(160, 120);
const probe = $('probe');
let brain = null, xrayOn = false, prev = null, heat = null, kindRGB = [];

function setupXray(map) {
  brain = map;
  kindRGB = map.kinds.map((k) => [1, 3, 5].map((i) => parseInt(k.color.slice(i, i + 2), 16)));
  prev = new Uint8Array(map.rows * 160);
  heat = new Float32Array(map.rows * 160);
  const legend = $('legend');
  for (const k of map.kinds) {
    if (!k.bytes) continue;
    const li = document.createElement('li');
    li.innerHTML = `<i style="background:${k.color}"></i><span>${k.label}</span><b>${k.bytes.toLocaleString('en')} B</b>`;
    legend.appendChild(li);
  }
}

function drawXray() {
  const d = ximg.data, n = brain.rows * 160;
  for (let i = 0; i < n; i++) {
    const v = cpu.ram[0x800 + ((i / 160) | 0) * 256 + (i % 160)];
    if (v !== prev[i]) { heat[i] = 1; prev[i] = v; }
    const kind = brain.kind.charCodeAt(i) - 48, c = kindRGB[kind];
    const lum = ((v & 3) + ((v >> 2) & 3) + ((v >> 4) & 3)) / 9;
    const k = brain.kinds[kind].key === 'free' ? 1 : 0.35 + 0.65 * lum;
    const h = heat[i];
    for (let j = 0; j < 3; j++) d[4 * i + j] = c[j] * k * (1 - h) + 255 * h;
    d[4 * i + 3] = 255;
    heat[i] = h * 0.8;
  }
  xctx.putImageData(ximg, 0, 0);
}

function setXray(on) {
  xrayOn = on && brain !== null;
  xray.hidden = !xrayOn;
  $('legendbox').hidden = !xrayOn;
  $('xraybtn').textContent = xrayOn ? 'Simple view' : 'Advanced view';
  $('xraybtn').classList.toggle('on', xrayOn);
  probe.textContent = xrayOn ? 'X-ray of the top of the screen: what each byte holds. Point at a pixel for details. White flashes: memory being written.' : '';
  if (xrayOn) { prev.fill(0); heat.fill(0); drawXray(); }
}
$('xraybtn').addEventListener('click', () => setXray(!xrayOn));

function probeAt(e) {
  const r = xray.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) / r.width * 160);
  const y = Math.floor((e.clientY - r.top) / r.height * 120);
  if (x < 0 || x >= 160 || y < 0 || y >= 120) return;
  if (y >= brain.rows) { probe.textContent = `line ${y}: the chat, drawn by the program`; return; }
  const i = y * 160 + x, addr = 0x800 + y * 256 + x;
  const what = brain.labels[brain.label[i]];
  probe.textContent = `line ${y}, address ${addr.toString(16).padStart(4, '0')}: ${what}`;
}
xray.addEventListener('pointermove', probeAt);
xray.addEventListener('pointerdown', probeAt);

async function boot() {
  const [rom, prog, info] = await Promise.all([
    fetch('rom/v4.rom').then((r) => r.arrayBuffer()),
    fetch('ttlm.gt1', { cache: 'no-cache' }).then((r) => r.arrayBuffer()),
    fetch('ttlm.json', { cache: 'no-cache' }).then((r) => r.json()),
  ]);
  const dv = new DataView(rom);
  for (let i = 0; i < cpu.rom.length; i++) cpu.rom[i] = dv.getUint16(2 * i);
  gt1 = new Uint8Array(prog);
  stateAddr = info.state;
  if (info.brain) setupXray(info.brain);
  for (let i = 0; i < cpu.ram.length; i++) cpu.ram[i] = (Math.random() * 256) | 0;
  cpu.reset();
  requestAnimationFrame(loop);
}
boot().catch((e) => { status.textContent = 'Could not start the Gigatron: ' + e.message; });
