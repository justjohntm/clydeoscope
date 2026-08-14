// Embeddable mount point for the landing-page visualiser tile.
//
// Self-contained: no bus, no server, no build step. Drops a WebGL canvas
// into `container`, auto-animates it via GentleEngine's synthetic driver,
// and wires up the guardrails a page-embedded tile needs that a full-screen
// pod display doesn't (off-screen pause, tab-hidden pause, reduced-motion,
// capped pointer/scroll influence).

'use strict';

import { GentleEngine } from './engine/GentleEngine.js';
import { GENTLE_VISUAL } from './config/gentle.visual.js';

// container: an element already sized/clipped by the caller (e.g. the
// circular .visualiser-frame) that the canvas should fill.
// Returns { destroy() } for cleanup, or null if WebGL isn't available (the
// caller's existing placeholder content is left untouched in that case).
export function mountVisualiserTile(container, options = {}) {
  const basePath = options.basePath || '';

  const canvas = document.createElement('canvas');
  canvas.className = 'visualiser-tile-canvas';
  canvas.setAttribute('aria-hidden', 'true');

  let engine;
  try {
    engine = new GentleEngine(canvas, withBasePath(GENTLE_VISUAL, basePath));
  } catch (e) {
    // No WebGL (or context creation failed): leave the caller's placeholder
    // in place rather than showing a broken canvas.
    console.warn('visualiser-tile: WebGL unavailable, leaving placeholder.', e);
    return null;
  }

  // Appended last so it stacks visually above the caller's placeholder
  // content; if WebGL setup fails above, we return before this line and the
  // placeholder is left as the only thing in the container.
  container.appendChild(canvas);

  // ---- reduced motion --------------------------------------------------
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  engine.setReducedMotion(motionQuery.matches);
  const onMotionChange = (e) => engine.setReducedMotion(e.matches);
  motionQuery.addEventListener ? motionQuery.addEventListener('change', onMotionChange)
    : motionQuery.addListener(onMotionChange);

  // ---- visibility guardrails: off-screen (IntersectionObserver) and ----
  // ---- tab-hidden (visibilitychange) both have to allow running --------
  let inView = false;
  let tabVisible = !document.hidden;
  function syncRunning() {
    if (inView && tabVisible) engine.start();
    else engine.stop();
  }

  const io = new IntersectionObserver((entries) => {
    inView = entries.some((e) => e.isIntersecting);
    syncRunning();
  }, { threshold: 0.05 });
  io.observe(container);

  function onVisibilityChange() {
    tabVisible = !document.hidden;
    syncRunning();
  }
  document.addEventListener('visibilitychange', onVisibilityChange);

  // ---- resize ------------------------------------------------------------
  const ro = new ResizeObserver(() => engine.sim.resize());
  ro.observe(container);

  // ---- pointer influence: soft, throttled, capped inside the engine -----
  let lastPointerT = 0;
  function onPointerMove(e) {
    const now = performance.now();
    if (now - lastPointerT < 140) return;
    lastPointerT = now;
    const rect = container.getBoundingClientRect();
    const px = e.clientX != null ? e.clientX : (e.touches && e.touches[0].clientX);
    const py = e.clientY != null ? e.clientY : (e.touches && e.touches[0].clientY);
    if (px == null || py == null) return;
    const x = (px - rect.left) / rect.width;
    const y = 1 - (py - rect.top) / rect.height; // sim's y is measured from the bottom
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    engine.addPointerInfluence(x, y, 0.6);
  }
  container.addEventListener('pointermove', onPointerMove, { passive: true });
  container.addEventListener('touchmove', onPointerMove, { passive: true });

  // ---- scroll influence: small decaying bump, not a position mapping ----
  let lastScrollY = window.scrollY;
  let scrollTicking = false;
  function onScroll() {
    if (scrollTicking) return;
    scrollTicking = true;
    requestAnimationFrame(() => {
      const dy = window.scrollY - lastScrollY;
      lastScrollY = window.scrollY;
      const normalized = Math.min(1, Math.abs(dy) / (window.innerHeight * 0.15));
      engine.bumpScroll(normalized);
      scrollTicking = false;
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });

  return {
    destroy() {
      engine.stop();
      io.disconnect();
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVisibilityChange);
      container.removeEventListener('pointermove', onPointerMove);
      container.removeEventListener('touchmove', onPointerMove);
      window.removeEventListener('scroll', onScroll);
      motionQuery.removeEventListener ? motionQuery.removeEventListener('change', onMotionChange)
        : motionQuery.removeListener(onMotionChange);
      canvas.remove();
    },
  };
}

function withBasePath(config, basePath) {
  if (!basePath) return config;
  const sources = {};
  for (const [cat, src] of Object.entries(config.sources)) sources[cat] = basePath + src;
  return Object.assign({}, config, { sources });
}
