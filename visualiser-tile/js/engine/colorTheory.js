// Colour theory for the fluid palette (Brief Section 2).
//
// Colour is used deliberately per phase, not as one fixed palette. Each phase
// declares a hue sweep + a colour *relationship* (analogous, split-complementary,
// etc.); saturation/brightness track arousal (beta) independently of hue; and a
// single reserved accent colour marks HRV coherence peaks. This module is pure
// colour maths — the per-phase numbers live in the experience visual config.

'use strict';

export function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
export function lerp(a, b, t) { return a + (b - a) * t; }

// Shortest-path hue interpolation around the 360deg wheel.
export function lerpHue(a, b, t) {
  let d = ((b - a) % 360 + 540) % 360 - 180;
  return (a + d * t + 360) % 360;
}

// HSL (h in degrees, s/l in 0..1) -> linear-ish RGB 0..1.
export function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360 / 360;
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return { r, g, b };
}

function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

// Resolve a "driver" value name to a channel number.
export function driverValue(driver, ch, progress) {
  switch (driver) {
    case 'alpha': return ch.alpha_norm;
    case 'theta': return ch.theta_norm;
    case 'beta': return ch.beta_norm;
    case 'coherence': return ch.hrv_coherence_norm;
    case 'progress': return progress;
    case 'inv_beta': return 1 - ch.beta_norm;
    default: return progress;
  }
}

// Interpolate a range { from, to, driver, invert } against channels + progress.
export function resolveRange(range, ch, progress) {
  if (typeof range === 'number') return range;
  let t = driverValue(range.driver || 'progress', ch, progress);
  if (range.invert) t = 1 - t;
  t = clamp01(t);
  return lerp(range.from, range.to, t);
}

// Build the current "flow colour" for a phase palette.
// palette: { hue:{from,to,driver,invert}, sat:{base,gain,driver}, light:{...} }
export function phaseColor(palette, ch, progress) {
  const h = resolveHue(palette.hue, ch, progress);
  const s = clamp01(resolveGain(palette.sat, ch, progress));
  const l = clamp01(resolveGain(palette.light, ch, progress));
  const rgb = hslToRgb(h, s, l);
  return { rgb, h, s, l };
}

function resolveHue(hue, ch, progress) {
  let t = driverValue(hue.driver || 'progress', ch, progress);
  if (hue.invert) t = 1 - t;
  return lerpHue(hue.from, hue.to, clamp01(t));
}

// A gain spec { base, gain, driver } -> base + gain*driver.
function resolveGain(spec, ch, progress) {
  if (typeof spec === 'number') return spec;
  const base = spec.base != null ? spec.base : 0;
  const gain = spec.gain != null ? spec.gain : 0;
  const d = driverValue(spec.driver || 'beta', ch, progress);
  return base + gain * d;
}

// The reserved accent colour for coherence peaks, scaled by peak intensity.
export function accentColor(accent, intensity = 1) {
  if (!accent) return { r: 1, g: 0.6, b: 0.15 };
  const l = clamp01((accent.l != null ? accent.l : 0.6) * (0.7 + 0.5 * intensity));
  return hslToRgb(accent.h, accent.s != null ? accent.s : 0.9, l);
}
