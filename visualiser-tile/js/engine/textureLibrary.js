// Texture/video input module (Visualiser 2.0, layer 1).
//
// One interface, two providers:
//   * Procedural (default this round): nature textures generated in-shader
//     (layered value-noise) to the brief's specs — moss, fern, running water,
//     lichen-covered bark, sparkle/bokeh. No external images.
//   * Image/video (the swap-in path for John's own park-walk footage):
//     loadImage()/loadVideo() replace a category through the exact same
//     getTexture() interface — no pipeline changes needed when real footage
//     arrives.
//
// Runs in the FluidSim's GL context, reusing its shader plumbing.

'use strict';

import { compileShader, Program, VERT_BASE } from '../fluid/fluidEngine.js';

// V2.2: 'bark' is scene content in its own right (the ring frame is gone);
// its procedural fallback reuses the lichen-bark recipe (index 3).
export const CATEGORIES = ['moss', 'fern', 'water', 'lichen', 'sparkle', 'bark'];
const CAT_INDEX = { moss: 0, fern: 1, water: 2, lichen: 3, sparkle: 4, bark: 3 };
// Water and sparkle animate (flow streaks / twinkle); the rest are static.
const ANIMATED = new Set(['water', 'sparkle']);

const TEX_SIZE = 512;
const ANim_INTERVAL_MS = 80; // ~12fps regeneration is plenty for slow textures

export class TextureLibrary {
  constructor(sim) {
    this.sim = sim;
    this.gl = sim.gl;
    const gl = this.gl;

    this.program = new Program(
      gl,
      compileShader(gl, gl.VERTEX_SHADER, VERT_BASE),
      compileShader(gl, gl.FRAGMENT_SHADER, FRAG_PROCTEX)
    );

    this.entries = {};
    for (const cat of CATEGORIES) {
      this.entries[cat] = this._makeTarget();
      this.entries[cat].mode = 'procedural';
      this._renderProcedural(cat, 0);
    }
    this._lastAnim = 0;
  }

  _makeTarget() {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_SIZE, TEX_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo, width: TEX_SIZE, height: TEX_SIZE };
  }

  _renderProcedural(cat, time) {
    const gl = this.gl;
    const e = this.entries[cat];
    this.program.bind();
    gl.uniform1f(this.program.uniforms.uCat, CAT_INDEX[cat]);
    gl.uniform1f(this.program.uniforms.uTime, time);
    gl.viewport(0, 0, e.width, e.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, e.fbo);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  // Called once per frame by the engine: regenerates the animated procedural
  // categories, and pulls fresh frames for any live video categories.
  update(timeSec) {
    const nowMs = timeSec * 1000;
    if (nowMs - this._lastAnim >= ANim_INTERVAL_MS) {
      this._lastAnim = nowMs;
      for (const cat of CATEGORIES) {
        const e = this.entries[cat];
        if (e.mode === 'procedural' && ANIMATED.has(cat)) this._renderProcedural(cat, timeSec);
      }
    }
    // Video categories: upload only when the video has actually advanced —
    // re-uploading an unchanged frame every rAF (60fps render vs ~30fps video)
    // is a full-frame GPU transfer for nothing and causes stalls on iGPUs.
    // The source clips are short (~5-10s), so `loop` produces a hard visual
    // cut each cycle — exactly a periodic "shunt". At each wrap we hold the
    // outgoing frame and crossfade it over the restarting video (~1.2s) on a
    // 2D canvas before upload.
    const gl = this.gl;
    const frameDt = Math.max(0, timeSec - (this._lastUpdateT || timeSec));
    this._lastUpdateT = timeSec;
    for (const cat of CATEGORIES) {
      const e = this.entries[cat];
      if (e.mode === 'video' && e.video && e.video.readyState >= 2) {
        const v = e.video;
        const t = v.currentTime;
        if (e.loopFade > 0) e.loopFade = Math.max(0, e.loopFade - frameDt / 1.2);
        if (t !== e._lastVideoTime && v.videoWidth) {
          if (!e.blendCanvas) {
            e.blendCanvas = document.createElement('canvas');
            e.blendCanvas.width = v.videoWidth; e.blendCanvas.height = v.videoHeight;
            e.blendCtx = e.blendCanvas.getContext('2d');
            e.prevCanvas = document.createElement('canvas');
            e.prevCanvas.width = v.videoWidth; e.prevCanvas.height = v.videoHeight;
            e.prevCtx = e.prevCanvas.getContext('2d');
          }
          if (e._lastVideoTime != null && t < e._lastVideoTime - 0.5) {
            // Looped: snapshot the final pre-wrap composite, start the fade.
            e.prevCtx.drawImage(e.blendCanvas, 0, 0);
            e.loopFade = 1;
          }
          e._lastVideoTime = t;
          const ctx = e.blendCtx;
          ctx.globalAlpha = 1;
          ctx.drawImage(v, 0, 0);
          if (e.loopFade > 0.01) {
            ctx.globalAlpha = e.loopFade;
            ctx.drawImage(e.prevCanvas, 0, 0);
            ctx.globalAlpha = 1;
          }
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, e.tex);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, e.blendCanvas);
        }
      }
    }
  }

  getTexture(cat) {
    const e = this.entries[cat] || this.entries.moss;
    return e.tex;
  }

  // ---- real-footage swap-in (same interface, later round) -----------------

  loadImage(cat, url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const gl = this.gl;
        const e = this.entries[cat] || (this.entries[cat] = this._makeTarget());
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, e.tex);
        // NPOT-safe: clamp + linear; seamless tiling is done in-shader via
        // mirror-repeat sampling, so wrap mode doesn't matter.
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        e.mode = 'image';
        resolve(e.tex);
      };
      img.onerror = () => reject(new Error('image load failed: ' + url));
      img.src = url;
    });
  }

  loadVideo(cat, url) {
    const video = document.createElement('video');
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.src = url;
    const e = this.entries[cat] || (this.entries[cat] = this._makeTarget());
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, e.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    e.mode = 'video';
    e.video = video;
    const p = video.play();
    if (p && p.catch) p.catch(() => {});
    return video;
  }

  // Reset a category back to the procedural placeholder.
  useProcedural(cat) {
    const e = this.entries[cat];
    if (!e) return;
    if (e.video) { try { e.video.pause(); } catch (err) {} e.video = null; }
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, e.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_SIZE, TEX_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    e.mode = 'procedural';
    this._renderProcedural(cat, 0);
  }
}

// ---------------------------------------------------------------------------
// Procedural nature textures. Layered value-noise ("Perlin-style") tuned per
// the brief's category descriptions. uCat selects the recipe.
// ---------------------------------------------------------------------------
const FRAG_PROCTEX = `
precision highp float;
varying vec2 vUv;
uniform float uCat;
uniform float uTime;

float hash21 (vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
vec2 hash22 (vec2 p) {
  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}
float vnoise (vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm (vec2 p) {
  float v = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    v += amp * vnoise(p);
    p = p * 2.03 + 17.1;
    amp *= 0.5;
  }
  return v;
}

// Moss: dense soft clumps, mid-to-dark green, yellow-green highlights, fine
// granular detail, no hard edges.
vec3 moss (vec2 uv) {
  float clump = fbm(uv * 6.0);
  float grain = fbm(uv * 42.0);
  float fine = vnoise(uv * 160.0);
  vec3 dark = vec3(0.045, 0.11, 0.04);
  vec3 mid = vec3(0.13, 0.30, 0.09);
  vec3 hi = vec3(0.45, 0.58, 0.16);
  vec3 c = mix(dark, mid, smoothstep(0.3, 0.75, clump));
  c = mix(c, hi, smoothstep(0.68, 0.9, clump * 0.65 + grain * 0.35) * 0.7);
  c *= 0.82 + 0.36 * fine; // granular sparkle without hard edges
  return c;
}

// Ferns: fronds radiating from a rib, serrated leaflets, bright green with
// warm gold at lit tips.
vec3 fern (vec2 uv) {
  vec3 c = vec3(0.03, 0.07, 0.035); // shaded undergrowth base
  for (int k = 0; k < 5; k++) {
    float fk = float(k);
    vec2 anchor = hash22(vec2(fk * 3.7, fk * 9.1));
    float ang = hash21(vec2(fk, 1.3)) * 6.28318;
    vec2 p = fract(uv - anchor) - 0.5;
    vec2 q = vec2(cos(ang) * p.x - sin(ang) * p.y, sin(ang) * p.x + cos(ang) * p.y);
    // Frond along +x from the anchor: taper, central rib, serrated edge.
    float along = q.x;
    if (along < 0.0 || along > 0.55) continue;
    float taper = 0.09 * (1.0 - along / 0.55);
    float serr = abs(fract(along * 26.0) - 0.5) * 2.0; // leaflet teeth
    float width = taper * (0.55 + 0.45 * serr);
    float leaf = smoothstep(width, width * 0.55, abs(q.y));
    float rib = smoothstep(0.008, 0.0, abs(q.y));
    float tip = smoothstep(0.3, 0.55, along);
    vec3 green = mix(vec3(0.10, 0.34, 0.10), vec3(0.30, 0.52, 0.14), serr);
    vec3 gold = vec3(0.72, 0.58, 0.2);
    vec3 fc = mix(green, gold, tip * 0.55) + rib * vec3(0.08, 0.1, 0.04);
    c = mix(c, fc, leaf * 0.85);
  }
  c *= 0.85 + 0.3 * fbm(uv * 30.0);
  return c;
}

// Running water: directional streaks + ripples, teal/white highlights over a
// darker blue-green base, foam specks. Animated.
vec3 water (vec2 uv) {
  vec2 flow = vec2(uTime * 0.11, uTime * 0.023);
  float streak = fbm(vec2(uv.x * 2.4, uv.y * 13.0) + flow * vec2(1.0, 6.0));
  float ridge = pow(1.0 - abs(streak * 2.0 - 1.0), 3.0);
  float swirl = fbm(uv * 4.0 + flow * 0.6);
  vec3 base = mix(vec3(0.012, 0.10, 0.13), vec3(0.03, 0.2, 0.24), swirl);
  vec3 c = base + ridge * vec3(0.28, 0.62, 0.62);
  float foam = step(0.955, vnoise(uv * 90.0 + flow * 14.0)) * ridge;
  c += foam * vec3(0.85, 0.95, 0.95);
  return c;
}

// Lichen-covered bark: rough vertical cracked bark, muted brown/grey, patchy
// teal-green lichen blotches with granular edges.
vec3 lichen (vec2 uv) {
  float rb = abs(fbm(uv * vec2(15.0, 2.6)) * 2.0 - 1.0); // vertical ridges
  float crack = smoothstep(0.0, 0.28, rb);
  vec3 bark = mix(vec3(0.055, 0.04, 0.03), mix(vec3(0.21, 0.16, 0.12), vec3(0.3, 0.28, 0.25), fbm(uv * 8.0)), crack);
  float patch = fbm(uv * 5.0 + 7.31);
  float lich = smoothstep(0.55, 0.66, patch + 0.14 * fbm(uv * 26.0));
  vec3 lcol = mix(vec3(0.16, 0.34, 0.26), vec3(0.34, 0.52, 0.34), fbm(uv * 40.0));
  return mix(bark, lcol, lich * 0.85);
}

// Sparkle: soft warm-gold/white bokeh points over a dark translucent base,
// irregular scatter, gentle twinkle. Animated.
vec3 sparkle (vec2 uv) {
  vec3 c = vec3(0.015, 0.02, 0.028);
  vec2 g = uv * 7.0;
  vec2 cell = floor(g);
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 cl = cell + vec2(float(x), float(y));
      vec2 pt = cl + hash22(cl);
      float sizeR = 0.12 + 0.3 * hash21(cl * 1.7);
      float d = length(g - pt);
      float tw = 0.55 + 0.45 * sin(uTime * 1.4 + hash21(cl) * 6.28318);
      float glowAmt = exp(-d * d / (sizeR * sizeR)) * tw;
      vec3 tint = mix(vec3(1.0, 0.82, 0.42), vec3(1.0, 0.97, 0.9), hash21(cl + 3.1));
      c += glowAmt * tint * 0.55;
    }
  }
  return c;
}

void main () {
  vec2 uv = vUv;
  vec3 c;
  if (uCat < 0.5) c = moss(uv);
  else if (uCat < 1.5) c = fern(uv);
  else if (uCat < 2.5) c = water(uv);
  else if (uCat < 3.5) c = lichen(uv);
  else c = sparkle(uv);
  gl_FragColor = vec4(c, 1.0);
}`;
