// Gentle, self-driving fork of Clydeoscope's CoreEngine for the landing-page
// visualiser tile.
//
// CoreEngine (the pod build) has exactly two entry points that carry
// biometric/session data in from the bus: setChannels(payload) and
// handleEvent(evt). Everything else — the fluid sim, splat placement,
// texture-library, symmetry easing, colour math — is plain rendering/maths
// with no data-source opinion. This fork keeps all of that and replaces the
// two bus entry points with a local synthetic driver (slow sine/pink wander)
// plus small, hard-capped pointer/scroll influence hooks. There is no bus,
// no socket.io, no session concept here at all.

'use strict';

import { FluidSim } from '../fluid/fluidEngine.js';
import { SplatScheduler, AmbientCurrents } from './splatField.js';
import { TextureLibrary, CATEGORIES } from './textureLibrary.js';
import { phaseColor, resolveRange, clamp01, lerp } from './colorTheory.js';

export class GentleEngine {
  constructor(canvas, visualConfig) {
    this.canvas = canvas;
    this.vc = visualConfig;
    this.phase = visualConfig.phase;

    this.sim = new FluidSim(canvas, {
      DENSITY_DISSIPATION: 0.6,
      // Small circular tile, not a full-screen pod display — a fraction of
      // the pod build's internal resolution is plenty and keeps this cheap
      // enough for mobile.
      SIM_RESOLUTION: 96,
      DYE_RESOLUTION: 512,
      BLOOM_RESOLUTION: 160,
    });
    this.scheduler = new SplatScheduler();
    this.currents = new AmbientCurrents(3);

    this.sym = { orderA: 2, orderB: 2, blend: 1, mix: 0, rot: 0, rotSpeed: 0 };
    this.pace = 1;

    this.texLib = new TextureLibrary(this.sim);
    this.sim.config.TEXTURE_MODE = 1;
    this.texW = {};
    for (const c of CATEGORIES) this.texW[c] = 0;
    this._slots = [null, null, null, null];
    this._loadRealSources(visualConfig.sources);

    // Synthetic "channels": a calm, slowly-wandering stand-in for the
    // biometric values the pod build would otherwise receive over the bus.
    // Kept mid-range and low-amplitude on purpose — this drives the same
    // GLOBAL_MAPPING ranges the pod uses, just dialed toward the gentle end.
    this.target = { alpha_norm: 0.5, theta_norm: 0.5, beta_norm: 0.4, hrv_coherence_norm: 0.45 };
    this.ch = Object.assign({}, this.target);
    this._synthT = Math.random() * 1000;

    this.phaseProgress = 0;
    this.paramOverrides = {};
    this.running = false;
    this.reducedMotion = false;

    this._raf = null;
    this._lastFrame = performance.now();
    this._sm = {};
    this._resizeAccum = 1;
    this._gusts = [];
    this._colSm = null;

    // Pointer influence: soft, capped, undirected nudges near the pointer.
    this._pointerCap = 0.22; // hard ceiling on gust strength (0..1 scale below)

    // Scroll influence: a decaying "energy" bumped by scroll events, eased
    // into a tight pace/rate multiplier so scrolling nudges the flow rather
    // than snapping it.
    this._scrollEnergy = 0;
    this._scrollMul = 1;

    this._onFrame = this._onFrame.bind(this);
  }

  start() {
    if (this._raf) return;
    this.running = true;
    this._lastFrame = performance.now();
    this._raf = requestAnimationFrame(this._onFrame);
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.running = false;
  }

  clear() { this.sim.clear(); }

  setReducedMotion(on) { this.reducedMotion = !!on; }

  // Soft, undirected nudge near a normalised (0..1, y from bottom) point.
  // Strength is clamped hard so repeated calls can never accumulate into
  // splat-drawing behaviour — this is meant to read as "the fluid noticing
  // you're there", not a paintbrush.
  addPointerInfluence(x, y, strength = 1) {
    if (this.reducedMotion) return;
    const s = clamp01(strength) * this._pointerCap;
    if (s <= 0.001) return;
    const ang = Math.random() * Math.PI * 2;
    const col = this._colSm || { r: 0.2, g: 0.5, b: 0.5 };
    const cfg = this.sim.config;
    this._spawnGust(
      clamp01(x), clamp01(y), Math.cos(ang), Math.sin(ang),
      cfg.SPLAT_FORCE * 0.0006 * s,
      { r: col.r * 0.4 * s, g: col.g * 0.4 * s, b: col.b * 0.4 * s },
      0.35, 0.5, 1.0, 1.8
    );
  }

  // Bump the decaying scroll-energy value (0..1) that eases into a tight
  // pace/rate multiplier. Call with a small normalised delta per scroll
  // event; energy decays back to 0 on its own each frame.
  bumpScroll(amount = 0.3) {
    if (this.reducedMotion) return;
    this._scrollEnergy = clamp01(this._scrollEnergy + Math.abs(amount));
  }

  _loadRealSources(sources) {
    if (!sources) return;
    for (const [cat, src] of Object.entries(sources)) {
      try {
        if (typeof src === 'string') this.texLib.loadImage(cat, src).catch(() => {});
      } catch (e) { /* keep procedural */ }
    }
  }

  _mappingFor(key) {
    const pm = this.phase.mapping && this.phase.mapping[key];
    return pm != null ? pm : this.vc.mapping[key];
  }

  _resolve(key, fallback) {
    if (this.paramOverrides[key] != null) return this.paramOverrides[key];
    const spec = this._mappingFor(key);
    if (spec == null) return fallback;
    if (typeof spec === 'number') return spec;
    return resolveRange(spec, this.ch, this.phaseProgress);
  }

  _resolveSmooth(key, fallback, dt, tau = 1.8) {
    const target = this._resolve(key, fallback);
    if (!Number.isFinite(this._sm[key])) { this._sm[key] = target; return target; }
    this._sm[key] += (target - this._sm[key]) * (1 - Math.exp(-dt / tau));
    return this._sm[key];
  }

  _spawnGust(x, y, dirX, dirY, force, color, radius, attack = 0.6, release = 1.5, dur = 2.4) {
    if (this._gusts.length >= 24) return;
    this._gusts.push({ x, y, dirX, dirY, force, color, radius, attack, release, dur, age: 0 });
  }

  _applyGusts(pdt) {
    for (let i = this._gusts.length - 1; i >= 0; i--) {
      const g = this._gusts[i];
      g.age += pdt;
      if (g.age >= g.dur) { this._gusts.splice(i, 1); continue; }
      const env = sstep(g.age / g.attack) * (1 - sstep((g.age - (g.dur - g.release)) / g.release));
      if (env <= 0) continue;
      const envInt = g.dur - 0.5 * g.attack - 0.5 * g.release;
      const k = env * pdt / Math.max(0.1, envInt);
      this.sim.splat(g.x, g.y, g.dirX * g.force * k, g.dirY * g.force * k,
        { r: g.color.r * k, g: g.color.g * k, b: g.color.b * k }, g.radius);
    }
  }

  // Slow sine + pink-noise wander, kept in a calm mid-range band. This is the
  // whole replacement for a bus payload: same shape of numbers (0..1 norms),
  // just generated locally instead of arriving from biometrics.
  _driveSynthetic(dt) {
    this._synthT += dt;
    const t = this._synthT;
    this.target.alpha_norm = 0.5 + 0.12 * Math.sin(t * 0.05);
    this.target.theta_norm = 0.5 + 0.1 * Math.sin(t * 0.037 + 1.3);
    this.target.beta_norm = 0.38 + 0.08 * Math.sin(t * 0.061 + 2.6);
    this.target.hrv_coherence_norm = 0.45 + 0.1 * Math.sin(t * 0.023 + 0.7);
  }

  _updateSymmetry(dt, phase) {
    const s = phase.symmetry || {};
    const targetOrder = s.order != null ? s.order : 2;
    if (targetOrder !== this.sym.orderB) {
      this.sym.orderA = this.sym.orderB;
      this.sym.orderB = targetOrder;
      this.sym.blend = 0;
    }
    this.sym.blend = Math.min(1, this.sym.blend + dt / 6);
    const targetMix = s.mix != null ? s.mix : 0;
    this.sym.mix = lerp(this.sym.mix, targetMix, 1 - Math.exp(-dt * 0.35));
    this.sym.rotSpeed = lerp(this.sym.rotSpeed, s.rotate || 0, 1 - Math.exp(-dt * 0.3));
    const t = this.sim.clock;
    const wobble = 0.006 * Math.sin(t * 0.11) + 0.0045 * Math.sin(t * 0.073 + 1.7);
    this.sym.rot += (this.sym.rotSpeed + wobble) * dt;

    const cfg = this.sim.config;
    cfg.SYM_ORDER_A = this.sym.orderA;
    cfg.SYM_ORDER_B = this.sym.orderB;
    cfg.SYM_ORDER_BLEND = this.sym.blend;
    cfg.SYM_MIX = this.sym.mix;
    cfg.SYM_ROT = this.sym.rot;
    cfg.SYM_MODE = s.mode === 'wedge' ? 1 : s.mode === 'mirror' ? 2 : s.mode === 'quad' ? 3 : 0;
  }

  _onFrame(now) {
    this._raf = requestAnimationFrame(this._onFrame);
    let dt = (now - this._lastFrame) / 1000;
    this._lastFrame = now;
    if (dt > 0.05) dt = 0.05;

    // Reduced motion: fall back to a near-still state rather than freezing
    // outright — the sim keeps rendering, but wander/splats/pointer input
    // are all dialed to (near) zero so nothing visibly moves.
    const motionScale = this.reducedMotion ? 0.06 : 1;

    if (!this.reducedMotion) this._driveSynthetic(dt);
    const k = 1 - Math.exp(-dt * 2.7);
    for (const key of Object.keys(this.target)) {
      this.ch[key] = lerp(this.ch[key], this.target[key], k);
    }

    this._resizeAccum += dt;
    if (this._resizeAccum >= 0.5) {
      this._resizeAccum = 0;
      this.sim.resize();
    }

    const phase = this.phase;

    // Scroll energy: decays on its own, eases into a tight pace/rate
    // multiplier so a scroll gesture nudges the flow rather than snapping it.
    this._scrollEnergy = Math.max(0, this._scrollEnergy - dt * 0.35);
    const scrollTargetMul = 1 + 0.16 * this._scrollEnergy;
    this._scrollMul = lerp(this._scrollMul, scrollTargetMul, 1 - Math.exp(-dt * 1.5));

    const targetPace = (phase.timeScale != null ? phase.timeScale : 1) * motionScale * this._scrollMul;
    this.pace = lerp(this.pace, targetPace, 1 - Math.exp(-dt * 0.4));
    const pdt = dt * this.pace;

    const cfg = this.sim.config;
    cfg.DENSITY_DISSIPATION = this._resolveSmooth('densityDissipation', 0.5, dt);
    cfg.VELOCITY_DISSIPATION = this._resolveSmooth('velocityDissipation', 0.8, dt);
    cfg.CURL = this._resolveSmooth('curl', 12, dt);
    cfg.PRESSURE = this._resolveSmooth('pressure', 0.8, dt);
    cfg.SPLAT_FORCE = this._resolveSmooth('splatForce', 3200, dt);
    cfg.BLOOM_INTENSITY = this._resolveSmooth('bloomIntensity', 0.6, dt);
    cfg.BLOOM_THRESHOLD = this._resolveSmooth('bloomThreshold', 0.4, dt);
    const splatRadius = this._resolveSmooth('splatRadius', 0.5, dt);

    this._updateSymmetry(dt, phase);

    cfg.WARP = this._resolveSmooth('warp', 0.1, dt);
    cfg.SWIRL = this._resolveSmooth('swirl', 0.012, dt);
    cfg.FLOW_GAIN = this._resolveSmooth('flowGain', 1.4, dt);
    cfg.FLOW_RESTORE = this._resolveSmooth('flowRestore', 0.965, dt);
    cfg.TEX_TILE = this._resolveSmooth('texTile', 1.4, dt);
    cfg.GLOW_GAIN = this._resolveSmooth('glowGain', 1.0, dt);
    cfg.GRADE_MIX = this._resolveSmooth('gradeMix', 0.32, dt);
    cfg.BLACK_LIFT = this._resolveSmooth('blackLift', 0.035, dt);

    this.texLib.update(this.sim.clock);
    const targets = phase.textures || {};
    const kw = 1 - Math.exp(-dt * 0.4);
    const active = [];
    for (const c of CATEGORIES) {
      this.texW[c] = lerp(this.texW[c], targets[c] || 0, kw);
      if (c !== 'sparkle' && this.texW[c] > 0.015) active.push(c);
    }
    for (let i = 0; i < 4; i++) {
      if (this._slots[i] && !active.includes(this._slots[i])) this._slots[i] = null;
    }
    for (const c of active) {
      if (this._slots.includes(c)) continue;
      const free = this._slots.indexOf(null);
      if (free >= 0) this._slots[free] = c;
    }
    this.sim.setSceneTextures(this._slots.map((c) =>
      c ? { tex: this.texLib.getTexture(c), weight: this.texW[c] } : null));
    this.sim.setSparkle(this.texLib.getTexture('sparkle'), this.texW.sparkle);

    this.sim.step(pdt);

    const colT = phaseColor(phase, this.ch, this.phaseProgress);
    if (!this._colSm) this._colSm = { r: colT.rgb.r, g: colT.rgb.g, b: colT.rgb.b };
    const ck = 1 - Math.exp(-dt / 4);
    this._colSm.r += (colT.rgb.r - this._colSm.r) * ck;
    this._colSm.g += (colT.rgb.g - this._colSm.g) * ck;
    this._colSm.b += (colT.rgb.b - this._colSm.b) * ck;
    const col = { rgb: this._colSm };
    const bright = this._resolveSmooth('dyeBrightness', 0.5, dt);

    const amb = phase.ambient;
    if (amb && amb.dyeRate > 0) {
      const flows = this.currents.update(pdt, amb.drift != null ? amb.drift : 0.08);
      for (const f of flows) {
        const gain = amb.dyeRate * pdt * f.strength * motionScale;
        const len = Math.hypot(f.dx, f.dy) || 1;
        const force = (amb.force != null ? amb.force : 30) * pdt * f.strength;
        const radius = (amb.radius != null ? amb.radius : splatRadius);
        this.sim.splat(
          f.x, f.y,
          (f.dx / len) * force, (f.dy / len) * force,
          { r: col.rgb.r * bright * gain, g: col.rgb.g * bright * gain, b: col.rgb.b * bright * gain },
          radius
        );
      }
    }

    let rate = (phase.splatRate != null ? phase.splatRate : 0.2) * motionScale;
    rate *= 0.85 + 0.3 * this.ch.beta_norm;
    const splats = this.scheduler.update(pdt, rate, 0);
    for (const s of splats) {
      const b = bright * s.strength;
      let { x, y } = s;
      const len = Math.hypot(s.dx, s.dy) || 1;
      const dirX = s.dx / len, dirY = s.dy / len;
      const radius = splatRadius * (0.7 + 0.6 * s.strength);
      this._spawnGust(x, y, dirX, dirY, cfg.SPLAT_FORCE * 0.0016 * s.strength,
        { r: col.rgb.r * b, g: col.rgb.g * b, b: col.rgb.b * b }, radius);
    }

    this._applyGusts(pdt);
    this.sim.render(null);
  }
}

function sstep(t) { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); }
