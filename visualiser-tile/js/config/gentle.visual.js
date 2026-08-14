// Visual config for the landing-page visualiser tile.
//
// Adapted from the pod build's recharge.v2.visual.js: same teal/gold grade,
// same kaleidoscope/texture-scene machinery, but collapsed to a single calm
// phase (no five-phase arc, no biometric mapping targets) and tuned toward
// the gentle end of that config's own ranges — lower splat rate, lower
// force, slower time scale. mapping ranges still resolve against
// GentleEngine's synthetic channel wander, so the parameters keep a slow
// "breathing" quality without needing real biometrics.

'use strict';

const MAPPING = {
  densityDissipation: { from: 1.9, to: 1.0, driver: 'theta' },
  velocityDissipation: { from: 0.5, to: 0.3, driver: 'theta' },
  curl: { from: 5, to: 14, driver: 'beta' },
  pressure: { from: 0.75, to: 0.9, driver: 'coherence' },
  splatRadius: { from: 0.45, to: 0.85, driver: 'coherence' },
  dyeBrightness: { from: 0.4, to: 0.7, driver: 'beta' },
  bloomIntensity: 0.6,
  bloomThreshold: 0.4,
  splatForce: 3200,

  warp: { from: 0.08, to: 0.14, driver: 'theta' },
  swirl: 0.012,
  flowGain: 1.4,
  flowRestore: 0.965,
  texTile: 1.4,
  glowGain: 1.0,

  gradeMix: 0.32,
  blackLift: 0.035,
};

export const GENTLE_VISUAL = {
  id: 'visualiser-tile',
  mode: 'texture',
  background: { r: 0, g: 0, b: 0 },

  // Only 4 of the 6 texture categories are backed by real stills; moss and
  // bark are left on TextureLibrary's procedural fallback (no download),
  // which keeps the tile's asset payload small.
  sources: {
    fern: 'textures/woodland/fern.webp',
    lichen: 'textures/woodland/lichen.avif',
    water: 'textures/water/water.avif',
    sparkle: 'textures/water/sparkle.avif',
  },

  mapping: MAPPING,

  phase: {
    hue: { from: 182, to: 195, driver: 'alpha' },
    sat: { base: 0.55, gain: 0.12, driver: 'beta' },
    light: { base: 0.38, gain: 0.1, driver: 'beta' },
    textures: { fern: 0.3, lichen: 0.25, water: 0.25, sparkle: 0.2 },
    symmetry: { order: 8, mix: 0.85, rotate: 0.006, mode: 'wedge' },
    ambient: { dyeRate: 0.1, force: 26, drift: 0.06 },
    splatRate: 0.22,
    timeScale: 0.7,
  },
};

export default GENTLE_VISUAL;
