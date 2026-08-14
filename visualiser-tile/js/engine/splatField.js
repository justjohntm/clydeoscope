// Splat placement + timing (Brief Section 3).
//
// Two ideas from the brief made concrete:
//   * Placement is structured variety, not a fixed point or a single line:
//     splat positions are drawn from a curl-noise (divergence-free) flow field,
//     so they drift like a current rather than teleporting.
//   * Timing + amplitude follow fractal / 1/f ("pink") correlated randomness via
//     the Voss-McCartney algorithm — small movements build into bigger ones over
//     time, the way HRV and neural activity actually behave, so it never reads as
//     jarring or mechanical.

'use strict';

// Voss-McCartney pink-noise generator -> correlated values in ~-1..1.
export class PinkNoise {
  constructor(octaves = 5) {
    this.octaves = octaves;
    this.values = new Array(octaves).fill(0).map(() => Math.random() * 2 - 1);
    this.counter = 0;
    this.max = octaves + 1;
  }
  next() {
    this.counter++;
    let sum = Math.random() * 2 - 1; // the always-updating white row
    let c = this.counter;
    for (let i = 0; i < this.octaves; i++) {
      if ((c & 1) === 0) {
        this.values[i] = Math.random() * 2 - 1;
      }
      c >>= 1;
      sum += this.values[i];
    }
    return sum / this.max; // ~ -1..1, pink-correlated
  }
}

// Cheap 2D value noise + curl for divergence-free placement drift.
function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function smooth(t) { return t * t * (3 - 2 * t); }
function valueNoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const a = hash2(xi, yi);
  const b = hash2(xi + 1, yi);
  const c = hash2(xi, yi + 1);
  const d = hash2(xi + 1, yi + 1);
  const u = smooth(xf), v = smooth(yf);
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}
// Curl of a scalar noise potential -> a divergence-free 2D direction.
function curlNoise(x, y) {
  const e = 0.01;
  const n1 = valueNoise(x, y + e);
  const n2 = valueNoise(x, y - e);
  const n3 = valueNoise(x + e, y);
  const n4 = valueNoise(x - e, y);
  return { x: (n1 - n2) / (2 * e), y: -(n3 - n4) / (2 * e) };
}

export class SplatScheduler {
  constructor(opts = {}) {
    this.pinkTime = new PinkNoise(6);   // modulates inter-splat interval
    this.pinkAmp = new PinkNoise(5);    // modulates amplitude
    this.pinkX = new PinkNoise(4);
    this.pinkY = new PinkNoise(4);
    // A slowly wandering anchor that walks the curl-noise field.
    this.anchor = { x: 0.5, y: 0.5 };
    this.fieldT = Math.random() * 1000;
    this.timeToNext = 0.5;
    this.clock = 0;
  }

  // Advance by dt seconds. `rate` = target splats/sec; `ambient` = 0..1 gentle
  // drift factor (raised during signal dropout). Returns an array of splats:
  // { x, y, dx, dy, strength } with x,y in 0..1 (y from bottom).
  update(dt, rate, ambient = 0) {
    this.clock += dt;
    this.fieldT += dt * 0.05;
    const out = [];
    if (rate <= 0 && ambient <= 0) return out;

    // Drift the anchor along the curl field so successive splats flow.
    const dir = curlNoise(this.anchor.x * 3 + this.fieldT, this.anchor.y * 3 - this.fieldT);
    this.anchor.x = wrap01(this.anchor.x + dir.x * dt * 0.15 + (this.pinkX.next() * 0.002));
    this.anchor.y = wrap01(this.anchor.y + dir.y * dt * 0.15 + (this.pinkY.next() * 0.002));

    this.timeToNext -= dt;
    // Pink-noise-jittered interval: base interval from rate, warped by pink noise
    // so bursts and lulls emerge with 1/f structure.
    const effRate = Math.max(0.02, rate + ambient * 0.15);
    while (this.timeToNext <= 0) {
      const pinkI = this.pinkTime.next();          // -1..1
      const baseInterval = 1 / effRate;
      const interval = baseInterval * (0.55 + 0.9 * (0.5 + 0.5 * pinkI));
      this.timeToNext += interval;

      // Position: anchor + a curl-flowed offset + small pink jitter.
      const off = curlNoise(this.anchor.x * 6 + this.fieldT * 2, this.anchor.y * 6);
      const jx = this.pinkX.next() * 0.12;
      const jy = this.pinkY.next() * 0.12;
      const x = wrap01(this.anchor.x + off.x * 0.08 + jx);
      const y = wrap01(this.anchor.y + off.y * 0.08 + jy);

      // Amplitude: pink-correlated so gentle motions accumulate into bigger ones.
      const amp = 0.5 + 0.5 * this.pinkAmp.next(); // 0..1
      const strength = 0.35 + 0.65 * amp;

      // Velocity impulse points along the local flow direction.
      const flow = curlNoise(x * 8 + this.fieldT, y * 8 - this.fieldT);
      const mag = 1.0 * (0.4 + 0.6 * amp);
      out.push({
        x, y,
        dx: flow.x * mag,
        dy: flow.y * mag,
        strength,
      });
    }
    return out;
  }
}

// Continuous ambient injection (V2 "Clydeoscopy" brief): instead of discrete
// trigger-and-decay splats, a few slow "currents" wander the curl-noise field
// and pour force + dye in smoothly every frame, with a slow sine swell per
// current. There is no visible injection moment — forms morph continuously,
// like weather systems resolving into one another.
export class AmbientCurrents {
  constructor(count = 3) {
    this.points = [];
    for (let i = 0; i < count; i++) {
      this.points.push({
        x: Math.random(),
        y: Math.random(),
        swellPhase: Math.random() * Math.PI * 2,
        t: Math.random() * 100,
      });
    }
    this.clock = 0;
  }

  // Advance by dt seconds; `drift` scales how fast currents wander.
  // Returns one entry per current: { x, y, dx, dy, strength } where dx/dy is
  // the local (unnormalised) flow direction and strength swells 0.1..1.
  update(dt, drift = 0.1) {
    this.clock += dt;
    const out = [];
    for (const p of this.points) {
      p.t += dt * 0.05;
      const d = curlNoise(p.x * 2.2 + p.t, p.y * 2.2 - p.t * 0.7);
      p.x = wrap01(p.x + d.x * dt * drift);
      p.y = wrap01(p.y + d.y * dt * drift);
      const swell = 0.55 + 0.45 * Math.sin(this.clock * 0.22 + p.swellPhase);
      out.push({ x: p.x, y: p.y, dx: d.x, dy: d.y, strength: Math.max(0.1, swell) });
    }
    return out;
  }
}

function wrap01(v) {
  v = v % 1;
  return v < 0 ? v + 1 : v;
}
