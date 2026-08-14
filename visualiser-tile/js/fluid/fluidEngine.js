// Clydeoscope fluid engine.
//
// A modular fork of Pavel Dobryakov's WebGL-Fluid-Simulation (MIT licensed,
// https://github.com/PavelDoGreat/WebGL-Fluid-Simulation). The GPU Navier-Stokes
// solver (advection, curl/vorticity, divergence, Jacobi pressure, gradient
// subtract, dye) is preserved; it has been wrapped in a class with an explicit
// `config` object and a splat API so an experience config can drive every
// parameter the brief maps to (dye dissipation, vorticity/curl, splat radius,
// pressure dissipation). Bloom and the pseudo-3D display shading are ported
// back in (both are pure rendering passes with no biofeedback meaning of their
// own) because they're what give the original demo its glowing, dimensional
// look rather than flat colour. Sunrays/sunlight were left out — bloom alone
// gets the "smokey glow" without the extra cost of a radial godray pass.
//
// Original work Copyright (c) 2017 Pavel Dobryakov, MIT License.

'use strict';

export const DEFAULT_CONFIG = {
  SIM_RESOLUTION: 128,
  DYE_RESOLUTION: 1024,
  DENSITY_DISSIPATION: 0.4,   // low => the canvas accumulates the whole session
  VELOCITY_DISSIPATION: 0.8,
  PRESSURE: 0.8,
  PRESSURE_ITERATIONS: 20,
  CURL: 18,                   // vorticity
  SPLAT_RADIUS: 0.25,
  SPLAT_FORCE: 6000,
  COLORFUL: false,
  PAUSED: false,
  BACK_COLOR: { r: 0, g: 0, b: 0 },
  BLOOM_ITERATIONS: 8,
  BLOOM_RESOLUTION: 256,
  BLOOM_INTENSITY: 0.75,
  BLOOM_THRESHOLD: 0.65,
  BLOOM_SOFT_KNEE: 0.6,
  // Kaleidoscope ("Clydeoscopy") symmetry pass — applied to the dye at render
  // time only; the simulation itself stays asymmetric. Two orders + a blend
  // let the caller crossfade between symmetry counts without a visible snap
  // (a fractional fold order would tear one wedge instead).
  SYM_ORDER_A: 2,
  SYM_ORDER_B: 2,
  SYM_ORDER_BLEND: 1, // 0 => order A, 1 => order B
  SYM_MIX: 0,         // 0 => raw fluid, 1 => fully kaleidoscoped
  SYM_ROT: 0,         // radians; spins which wedge of the fluid is sampled
  // 0 = "Clydeoscopy" composite (max-blend of N rotated+mirrored copies; right
  // for sparse dye), 1 = classic radial wedge fold (primary V2 mode),
  // 2 = vertical axis mirror ("Rorschach/moth", V2.2), 3 = quadrant mirror.
  SYM_MODE: 0,

  // ---- Visualiser 2.0: texture scene + displacement + grade ---------------
  TEXTURE_MODE: 0,     // 1 => kaleido/bloom/display read the textured scene, not the dye
  FLOW_GAIN: 1.6,      // velocity -> accumulated UV displacement
  FLOW_RESTORE: 0.965, // per-step decay toward identity (stops infinite smearing)
  WARP: 0.3,           // how far the displacement field bends texture sampling
  SWIRL: 0.04,         // base spiral speed (rad/s, stronger near centre)
  TEX_TILE: 1.5,       // texture tiling scale in the scene pass
  GLOW_GAIN: 1.1,      // how much the dye reads as bioluminescent glow over texture
  GRADE_MIX: 0,        // colour-grade amount (V2.2: naturalistic base + accents)
  BLACK_LIFT: 0.035,   // crushed blacks
};

export class FluidSim {
  constructor(canvas, config = {}) {
    this.canvas = canvas;
    this.config = Object.assign({}, DEFAULT_CONFIG, config);
    this.splatStack = [];

    const { gl, ext } = getWebGLContext(canvas);
    this.gl = gl;
    this.ext = ext;
    if (!ext.supportLinearFiltering) {
      this.config.DYE_RESOLUTION = 512;
    }

    this._initShaders();
    this._initBlit();
    // Visualiser 2.0 state: scene textures (from TextureLibrary or, later,
    // real footage via the same interface), bark ring texture, sim clock.
    this.sceneTex = [];   // [{ tex, weight }] up to 4 (null holes keep slots stable)
    this.sparkleTex = null;   // dedicated additive glint layer (not a blend slot)
    this.sparkleWeight = 0;
    this.ringTex = null;
    this.clock = 0;
    this.swirlPhase = 0;  // integrated spiral angle (rad) — see step()
    this.dye = null;
    this.velocity = null;
    this.divergence = null;
    this.curlFbo = null;
    this.pressure = null;
    this.initFramebuffers();

    this.lastUpdate = Date.now();
  }

  // ---- shader / program setup ----------------------------------------------

  _initShaders() {
    const gl = this.gl;
    const baseVertex = compileShader(gl, gl.VERTEX_SHADER, VERT_BASE);

    this.copyProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_COPY));
    this.clearProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_CLEAR));
    this.splatProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SPLAT));
    this.advectionProgram = new Program(
      gl, baseVertex,
      compileShader(gl, gl.FRAGMENT_SHADER, FRAG_ADVECTION,
        this.ext.supportLinearFiltering ? null : ['MANUAL_FILTERING'])
    );
    this.divergenceProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_DIVERGENCE));
    this.curlProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_CURL));
    this.vorticityProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_VORTICITY));
    this.pressureProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_PRESSURE));
    this.gradienSubtractProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_GRADIENT_SUBTRACT));
    this.displayProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_DISPLAY));
    this.bloomPrefilterProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_BLOOM_PREFILTER));
    this.bloomBlurProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_BLOOM_BLUR));
    this.bloomFinalProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_BLOOM_FINAL));
    this.symmetryProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_KALEIDO));
    // Visualiser 2.0 passes.
    this.flowProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_FLOWSTEP));
    this.sceneProgram = new Program(gl, baseVertex, compileShader(gl, gl.FRAGMENT_SHADER, FRAG_TEXSCENE));
  }

  _initBlit() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    this.blit = (target, clear = false) => {
      if (target == null) {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      } else {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      }
      if (clear) {
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    };
  }

  // ---- framebuffers --------------------------------------------------------

  getResolution(resolution) {
    const gl = this.gl;
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1 / aspectRatio;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);
    if (gl.drawingBufferWidth > gl.drawingBufferHeight) return { width: max, height: min };
    return { width: min, height: max };
  }

  initFramebuffers() {
    const gl = this.gl;
    const ext = this.ext;
    const simRes = this.getResolution(this.config.SIM_RESOLUTION);
    const dyeRes = this.getResolution(this.config.DYE_RESOLUTION);
    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const rg = ext.formatRG;
    const r = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    gl.disable(gl.BLEND);

    this.dye = createDoubleFBO(gl, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);
    // Symmetrized copy of the scene, produced each frame by the kaleidoscope
    // pass; bloom and display shading both read from this so glow/lighting
    // stay consistent with the folded geometry. Full dye resolution: the
    // content is now real footage whose bark/leaf/ripple detail must survive
    // to the display pass — and the V2 symmetry modes (wedge/mirror/quad)
    // read only 1-2 samples per pixel, so full res stays cheap. (The
    // 12-sample composite mode is the V1 dye path only.)
    this.symmetric = createFBO(gl, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);
    this.velocity = createDoubleFBO(gl, simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filtering);
    this.divergence = createFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    this.curlFbo = createFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    this.pressure = createDoubleFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);

    // V2: accumulated UV-displacement field (advected by the velocity field,
    // decaying toward identity) + the textured scene it feeds.
    this.flow = createDoubleFBO(gl, simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filtering);
    this.scene = createFBO(gl, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);

    this._initBloomFramebuffers();
  }

  // A small mip chain: prefilter bright areas, blur-downsample, then
  // blur-upsample with additive blending back to full size. Same structure as
  // the original demo's bloom pass, at a lower, fixed resolution for cost.
  _initBloomFramebuffers() {
    const gl = this.gl;
    const ext = this.ext;
    const res = this.getResolution(this.config.BLOOM_RESOLUTION);
    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    this.bloom = createFBO(gl, res.width, res.height, rgba.internalFormat, rgba.format, texType, filtering);

    this.bloomFramebuffers = [];
    let width = res.width, height = res.height;
    for (let i = 0; i < this.config.BLOOM_ITERATIONS; i++) {
      width >>= 1; height >>= 1;
      if (width < 2 || height < 2) break;
      this.bloomFramebuffers.push(createFBO(gl, width, height, rgba.internalFormat, rgba.format, texType, filtering));
    }
  }

  resize() {
    const canvas = this.canvas;
    const w = scaleByPixelRatio(canvas.clientWidth || window.innerWidth);
    const h = scaleByPixelRatio(canvas.clientHeight || window.innerHeight);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      this.initFramebuffers();
    }
  }

  // ---- simulation step -----------------------------------------------------

  step(dt) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    const velocity = this.velocity;
    const config = this.config;
    this.clock += dt;
    // Integrate the spiral angle here rather than letting the shader compute
    // rate × absolute clock: with the latter, any change to SWIRL (phase
    // transition, operator slider, biometric mapping) rotates the whole scene
    // by delta × elapsed-seconds in a single frame — a snap that grows worse
    // the longer the session runs.
    this.swirlPhase += config.SWIRL * dt;

    // Curl.
    this.curlProgram.bind();
    gl.uniform2f(this.curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(this.curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    this.blit(this.curlFbo);

    // Vorticity (applies curl force).
    this.vorticityProgram.bind();
    gl.uniform2f(this.vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(this.vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(this.vorticityProgram.uniforms.uCurl, this.curlFbo.attach(1));
    gl.uniform1f(this.vorticityProgram.uniforms.curl, config.CURL);
    gl.uniform1f(this.vorticityProgram.uniforms.dt, dt);
    this.blit(velocity.write);
    velocity.swap();

    // Divergence.
    this.divergenceProgram.bind();
    gl.uniform2f(this.divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(this.divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    this.blit(this.divergence);

    // Clear pressure with decay.
    this.clearProgram.bind();
    gl.uniform1i(this.clearProgram.uniforms.uTexture, this.pressure.read.attach(0));
    gl.uniform1f(this.clearProgram.uniforms.value, config.PRESSURE);
    this.blit(this.pressure.write);
    this.pressure.swap();

    // Jacobi pressure solve.
    this.pressureProgram.bind();
    gl.uniform2f(this.pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(this.pressureProgram.uniforms.uDivergence, this.divergence.attach(0));
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(this.pressureProgram.uniforms.uPressure, this.pressure.read.attach(1));
      this.blit(this.pressure.write);
      this.pressure.swap();
    }

    // Gradient subtract.
    this.gradienSubtractProgram.bind();
    gl.uniform2f(this.gradienSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(this.gradienSubtractProgram.uniforms.uPressure, this.pressure.read.attach(0));
    gl.uniform1i(this.gradienSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    this.blit(velocity.write);
    velocity.swap();

    // Advect velocity.
    this.advectionProgram.bind();
    gl.uniform2f(this.advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!this.ext.supportLinearFiltering) {
      gl.uniform2f(this.advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    }
    gl.uniform1i(this.advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(this.advectionProgram.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(this.advectionProgram.uniforms.dt, dt);
    gl.uniform1f(this.advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    this.blit(velocity.write);
    velocity.swap();

    // Advect dye.
    if (!this.ext.supportLinearFiltering) {
      gl.uniform2f(this.advectionProgram.uniforms.dyeTexelSize, this.dye.texelSizeX, this.dye.texelSizeY);
    }
    gl.uniform1i(this.advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(this.advectionProgram.uniforms.uSource, this.dye.read.attach(1));
    gl.uniform1f(this.advectionProgram.uniforms.dissipation, config.DENSITY_DISSIPATION);
    this.blit(this.dye.write);
    this.dye.swap();

    // V2 displacement field: carry the accumulated UV offset along the flow,
    // keep injecting current velocity into it, decay toward identity so the
    // warp stays organic rather than smearing without limit.
    if (config.TEXTURE_MODE) {
      this.flowProgram.bind();
      gl.uniform2f(this.flowProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
      gl.uniform1i(this.flowProgram.uniforms.uVelocity, velocity.read.attach(0));
      gl.uniform1i(this.flowProgram.uniforms.uFlow, this.flow.read.attach(1));
      gl.uniform1f(this.flowProgram.uniforms.dt, dt);
      gl.uniform1f(this.flowProgram.uniforms.gain, config.FLOW_GAIN);
      gl.uniform1f(this.flowProgram.uniforms.restore, config.FLOW_RESTORE);
      this.blit(this.flow.write);
      this.flow.swap();
    }
  }

  // ---- rendering -----------------------------------------------------------

  // Provide the scene textures ([{ tex, weight }], max 4). Works identically
  // whether they come from the procedural TextureLibrary or real footage.
  // (V2.2: the bark ring frame is gone — bark is now ordinary scene content.)
  setSceneTextures(list) { this.sceneTex = (list || []).slice(0, 4); }

  // Sparkle is not one of the 4 blend slots: its dark base would only dim
  // whatever it averages with. It rides on top as additive glints.
  setSparkle(tex, weight) { this.sparkleTex = tex; this.sparkleWeight = weight || 0; }

  render(target = null) {
    const gl = this.gl;
    const cfg = this.config;

    // V2: build the textured scene (photo content sampled through the
    // displacement field, dye as bioluminescent glow) and feed THAT to the
    // kaleidoscope; V1 path feeds the dye directly.
    let symSource = this.dye.read;
    if (cfg.TEXTURE_MODE && this.sceneTex.some((e) => e)) {
      this._renderScene();
      symSource = this.scene;
    }
    this._applySymmetry(symSource);
    this._applyBloom(this.symmetric, this.bloom);

    gl.disable(gl.BLEND);
    this.displayProgram.bind();
    const u = this.displayProgram.uniforms;
    gl.uniform2f(u.texelSize, this.symmetric.texelSizeX, this.symmetric.texelSizeY);
    gl.uniform1i(u.uTexture, this.symmetric.attach(0));
    gl.uniform1i(u.uBloom, this.bloom.attach(1));
    gl.uniform1f(u.uTime, this.clock);
    // Grade (zero-mix by default => V1 output unchanged). V2.2: full-screen,
    // edge-to-edge — no ring, no vignette.
    gl.uniform1f(u.uGradeMix, cfg.GRADE_MIX);
    gl.uniform1f(u.uBlackLift, cfg.BLACK_LIFT);
    this.blit(target);
  }

  // Textured scene pass: spiral-swirled, displacement-warped blend of up to 4
  // nature textures, with the dye layered in as bioluminescent glow.
  _renderScene() {
    const gl = this.gl;
    const cfg = this.config;
    gl.disable(gl.BLEND);
    this.sceneProgram.bind();
    const u = this.sceneProgram.uniforms;
    const w = [0, 0, 0, 0];
    const fallbackEntry = this.sceneTex.find((e) => e);
    const fallbackTex = fallbackEntry ? fallbackEntry.tex : null;
    for (let i = 0; i < 4; i++) {
      const entry = this.sceneTex[i];
      gl.activeTexture(gl.TEXTURE3 + i);
      gl.bindTexture(gl.TEXTURE_2D, entry ? entry.tex : fallbackTex);
      gl.uniform1i(u['uTex' + i], 3 + i);
      w[i] = entry ? entry.weight : 0;
    }
    const wsum = w[0] + w[1] + w[2] + w[3] || 1;
    gl.uniform4f(u.uW, w[0] / wsum, w[1] / wsum, w[2] / wsum, w[3] / wsum);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, this.sparkleTex || fallbackTex);
    gl.uniform1i(u.uTexS, 7);
    gl.uniform1f(u.uSparkleW, this.sparkleTex ? this.sparkleWeight : 0);
    gl.uniform1i(u.uFlow, this.flow.read.attach(0));
    gl.uniform1i(u.uDye, this.dye.read.attach(1));
    gl.uniform1f(u.uTime, this.clock);
    gl.uniform1f(u.uSwirlPhase, this.swirlPhase);
    gl.uniform1f(u.uWarp, cfg.WARP);
    gl.uniform1f(u.uTile, cfg.TEX_TILE);
    gl.uniform1f(u.uGlow, cfg.GLOW_GAIN);
    gl.uniform1f(u.uAspect, this.canvas.width / Math.max(1, this.canvas.height));
    this.blit(this.scene);
  }

  // Kaleidoscope fold of the dye texture (render-time only; the sim stays
  // untouched). Runs even at SYM_MIX 0 (it degenerates to a copy) so the rest
  // of the render chain always reads one place.
  _applySymmetry(source) {
    const gl = this.gl;
    const c = this.config;
    gl.disable(gl.BLEND);
    this.symmetryProgram.bind();
    gl.uniform1f(this.symmetryProgram.uniforms.uMode, c.SYM_MODE || 0);
    gl.uniform1f(this.symmetryProgram.uniforms.uTime, this.clock);
    gl.uniform1i(this.symmetryProgram.uniforms.uTexture, (source || this.dye.read).attach(0));
    gl.uniform1f(this.symmetryProgram.uniforms.uOrderA, Math.max(1, c.SYM_ORDER_A));
    gl.uniform1f(this.symmetryProgram.uniforms.uOrderB, Math.max(1, c.SYM_ORDER_B));
    gl.uniform1f(this.symmetryProgram.uniforms.uOrderBlend, c.SYM_ORDER_BLEND);
    gl.uniform1f(this.symmetryProgram.uniforms.uMix, c.SYM_MIX);
    gl.uniform1f(this.symmetryProgram.uniforms.uRot, c.SYM_ROT);
    gl.uniform1f(this.symmetryProgram.uniforms.uAspect, this.canvas.width / Math.max(1, this.canvas.height));
    this.blit(this.symmetric);
  }

  // Bright-pass -> downsample-blur chain -> additive upsample -> combine.
  // Ported from the original demo's bloom pass (pure rendering, no data
  // meaning) so highlights glow instead of reading as flat colour.
  _applyBloom(source, destination) {
    if (!this.bloomFramebuffers || this.bloomFramebuffers.length < 2) return;
    const gl = this.gl;
    const config = this.config;
    let last = destination;

    gl.disable(gl.BLEND);
    this.bloomPrefilterProgram.bind();
    const knee = config.BLOOM_THRESHOLD * config.BLOOM_SOFT_KNEE + 0.0001;
    const curve0 = config.BLOOM_THRESHOLD - knee;
    const curve1 = knee * 2;
    const curve2 = 0.25 / knee;
    gl.uniform3f(this.bloomPrefilterProgram.uniforms.curve, curve0, curve1, curve2);
    gl.uniform1f(this.bloomPrefilterProgram.uniforms.threshold, config.BLOOM_THRESHOLD);
    gl.uniform1i(this.bloomPrefilterProgram.uniforms.uTexture, source.attach(0));
    this.blit(last);

    this.bloomBlurProgram.bind();
    for (let i = 0; i < this.bloomFramebuffers.length; i++) {
      const dest = this.bloomFramebuffers[i];
      gl.uniform2f(this.bloomBlurProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
      gl.uniform1i(this.bloomBlurProgram.uniforms.uTexture, last.attach(0));
      this.blit(dest);
      last = dest;
    }

    gl.blendFunc(gl.ONE, gl.ONE);
    gl.enable(gl.BLEND);
    for (let i = this.bloomFramebuffers.length - 2; i >= 0; i--) {
      const baseTex = this.bloomFramebuffers[i];
      gl.uniform2f(this.bloomBlurProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
      gl.uniform1i(this.bloomBlurProgram.uniforms.uTexture, last.attach(0));
      this.blit(baseTex);
      last = baseTex;
    }

    gl.disable(gl.BLEND);
    this.bloomFinalProgram.bind();
    gl.uniform2f(this.bloomFinalProgram.uniforms.texelSize, last.texelSizeX, last.texelSizeY);
    gl.uniform1i(this.bloomFinalProgram.uniforms.uTexture, last.attach(0));
    gl.uniform1f(this.bloomFinalProgram.uniforms.intensity, config.BLOOM_INTENSITY);
    this.blit(destination);
  }

  // ---- splats --------------------------------------------------------------

  // x, y in 0..1 (y measured from bottom). dx, dy are velocity impulses.
  // color is {r,g,b} in 0..1 (can exceed 1 for brighter dye).
  splat(x, y, dx, dy, color, radius) {
    const gl = this.gl;
    const r = radius != null ? radius : this.config.SPLAT_RADIUS;
    this.splatProgram.bind();
    gl.uniform1i(this.splatProgram.uniforms.uTarget, this.velocity.read.attach(0));
    gl.uniform1f(this.splatProgram.uniforms.aspectRatio, this.canvas.width / this.canvas.height);
    gl.uniform2f(this.splatProgram.uniforms.point, x, y);
    gl.uniform3f(this.splatProgram.uniforms.color, dx, dy, 0);
    gl.uniform1f(this.splatProgram.uniforms.radius, correctRadius(r / 100, this.canvas));
    this.blit(this.velocity.write);
    this.velocity.swap();

    gl.uniform1i(this.splatProgram.uniforms.uTarget, this.dye.read.attach(0));
    gl.uniform3f(this.splatProgram.uniforms.color, color.r, color.g, color.b);
    this.blit(this.dye.write);
    this.dye.swap();
  }

  // Clear all dye (used when starting fresh).
  clear() {
    const gl = this.gl;
    this.clearProgram.bind();
    gl.uniform1i(this.clearProgram.uniforms.uTexture, this.dye.read.attach(0));
    gl.uniform1f(this.clearProgram.uniforms.value, 0);
    this.blit(this.dye.write);
    this.dye.swap();
  }

  // Export the current framebuffer as a PNG data URL (the "living image").
  captureImage(type = 'image/png') {
    // Ensure the latest frame is present in the drawing buffer.
    this.render(null);
    return this.canvas.toDataURL(type);
  }
}

// ---- WebGL helpers (ported from the original) ------------------------------

function getWebGLContext(canvas) {
  const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: true };
  let gl = canvas.getContext('webgl2', params);
  const isWebGL2 = !!gl;
  if (!isWebGL2) {
    gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);
  }
  let halfFloat;
  let supportLinearFiltering;
  if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float');
    supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
  } else {
    halfFloat = gl.getExtension('OES_texture_half_float');
    supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
  }
  gl.clearColor(0.0, 0.0, 0.0, 1.0);

  const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : (halfFloat && halfFloat.HALF_FLOAT_OES);
  let formatRGBA, formatRG, formatR;
  if (isWebGL2) {
    formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
    formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
    formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
  } else {
    formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
  }

  return {
    gl,
    ext: { formatRGBA, formatRG, formatR, halfFloatTexType, supportLinearFiltering },
  };
}

function getSupportedFormat(gl, internalFormat, format, type) {
  if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
    switch (internalFormat) {
      case gl.R16F: return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
      case gl.RG16F: return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
      default: return null;
    }
  }
  return { internalFormat, format };
}

function supportRenderTextureFormat(gl, internalFormat, format, type) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  return status === gl.FRAMEBUFFER_COMPLETE;
}

class Program {
  constructor(gl, vertexShader, fragmentShader) {
    this.gl = gl;
    this.uniforms = {};
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(this.program));
    }
    const count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(this.program, i).name;
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
  }
  bind() { this.gl.useProgram(this.program); }
}

function compileShader(gl, type, source, keywords) {
  source = addKeywords(source, keywords);
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader), source);
  }
  return shader;
}

function addKeywords(source, keywords) {
  if (!keywords) return source;
  let prefix = '';
  keywords.forEach((k) => { prefix += '#define ' + k + '\n'; });
  return prefix + source;
}

function createFBO(gl, w, h, internalFormat, format, type, param) {
  gl.activeTexture(gl.TEXTURE0);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.viewport(0, 0, w, h);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const texelSizeX = 1.0 / w;
  const texelSizeY = 1.0 / h;
  return {
    texture, fbo, width: w, height: h, texelSizeX, texelSizeY,
    attach(id) {
      gl.activeTexture(gl.TEXTURE0 + id);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      return id;
    },
  };
}

function createDoubleFBO(gl, w, h, internalFormat, format, type, param) {
  let fbo1 = createFBO(gl, w, h, internalFormat, format, type, param);
  let fbo2 = createFBO(gl, w, h, internalFormat, format, type, param);
  return {
    width: w, height: h, texelSizeX: fbo1.texelSizeX, texelSizeY: fbo1.texelSizeY,
    get read() { return fbo1; },
    set read(v) { fbo1 = v; },
    get write() { return fbo2; },
    set write(v) { fbo2 = v; },
    swap() { const t = fbo1; fbo1 = fbo2; fbo2 = t; },
  };
}

function scaleByPixelRatio(input) {
  const pixelRatio = window.devicePixelRatio || 1;
  return Math.floor(input * pixelRatio);
}

function correctRadius(radius, canvas) {
  const aspectRatio = canvas.width / canvas.height;
  return aspectRatio > 1 ? radius * aspectRatio : radius;
}

// ---- shaders ---------------------------------------------------------------

const VERT_BASE = `
precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAG_COPY = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
uniform sampler2D uTexture;
void main () { gl_FragColor = texture2D(uTexture, vUv); }`;

const FRAG_CLEAR = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`;

const FRAG_DISPLAY = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
uniform sampler2D uBloom;
uniform vec2 texelSize;
uniform float uGradeMix;
uniform float uBlackLift;
uniform float uTime;

float dhash21 (vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float dnoise (vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(dhash21(i), dhash21(i + vec2(1.0, 0.0)), f.x),
             mix(dhash21(i + vec2(0.0, 1.0)), dhash21(i + vec2(1.0, 1.0)), f.x), f.y);
}

void main () {
  vec3 c = texture2D(uTexture, vUv).rgb;

  // Pseudo-3D shading: treat dye density as a height field and light it, so
  // the fluid reads as dimensional smoke rather than a flat colour wash.
  vec3 lc = texture2D(uTexture, vL).rgb;
  vec3 rc = texture2D(uTexture, vR).rgb;
  vec3 tc = texture2D(uTexture, vT).rgb;
  vec3 bc = texture2D(uTexture, vB).rgb;
  float dx = length(rc) - length(lc);
  float dy = length(tc) - length(bc);
  vec3 n = normalize(vec3(dx, dy, length(texelSize)));
  vec3 l = vec3(0.0, 0.0, 1.0);
  float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
  c *= diffuse;

  // Bloom breathes across the frame: a slow-drifting spatial noise varies the
  // glow density (0.85..1.15) so light pools and recedes organically instead
  // of glowing uniformly ("alive, slightly unpredictable" — not mechanical).
  float bn = dnoise(vUv * 2.4 + vec2(uTime * 0.011, uTime * 0.007));
  vec3 bloom = texture2D(uBloom, vUv).rgb * (0.85 + 0.3 * bn);
  vec3 hdr = c + bloom;

  // Reinhard tonemap: as accumulated density/bloom climbs past 1, compress
  // toward white gracefully instead of hard-clipping the whole frame to a
  // flat wash. Keeps hot spots glowing while colour further out stays legible.
  vec3 mapped = hdr / (1.0 + hdr);

  // ---- V2.2 grade: naturalistic base with targeted accents ----
  // The footage's real colour survives; the grade layers on top of it:
  // shadows lean deep teal (with an indigo-purple floor in the darkest band —
  // the reference's "purple pockets"), genuine highlights warm toward gold.
  // No global duotone override, no false-colour wash.
  if (uGradeMix > 0.001) {
    float lum = dot(mapped, vec3(0.299, 0.587, 0.114));
    vec3 indigo = vec3(0.2, 0.12, 0.34);
    vec3 teal = vec3(0.02, 0.42, 0.46);
    vec3 gold = vec3(1.05, 0.78, 0.32);
    vec3 shadowTint = mix(indigo, teal, smoothstep(0.03, 0.3, lum));
    vec3 duo = mix(shadowTint * (pow(lum, 0.72) * 2.1), gold * (pow(lum, 0.8) * 1.7), smoothstep(0.45, 0.8, lum));
    mapped = mix(mapped, duo, uGradeMix);
    // Crushed blacks.
    mapped = max(mapped - uBlackLift, 0.0) / (1.0 - uBlackLift);
    // Gentle S-curve so real-texture detail stays crisp through blend+tonemap.
    mapped = clamp((mapped - 0.5) * 1.16 + 0.5, 0.0, 1.0);
  }

  // Opaque output. (The old alpha=max(channel) made dark pixels near-
  // transparent: fine over the black page, but every captured PNG — including
  // the session "living images" — composited to a washed-out pastel on white.)
  gl_FragColor = vec4(mapped, 1.0);
}`;

// V2: accumulated displacement. Advect the existing offset along the velocity
// field (so warp travels WITH the fluid), inject current velocity, decay toward
// zero. The scene pass then samples textures at uv + offset*WARP.
const FRAG_FLOWSTEP = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uFlow;
uniform vec2 texelSize;
uniform float dt;
uniform float gain;
uniform float restore;
void main () {
  vec2 vel = texture2D(uVelocity, vUv).xy;
  vec2 coord = vUv - dt * vel * texelSize;
  vec2 f = texture2D(uFlow, coord).xy * restore + vel * dt * gain * texelSize.x;
  gl_FragColor = vec4(f, 0.0, 1.0);
}`;

// V2: the textured scene. Spiral swirl (stronger near centre) + displacement
// warp bend the sampling of the nature textures; the dye field rides on top
// as bioluminescent glow so the fluid still visibly "lives".
//
// V2.3 texture mixing: organic patches instead of a uniform 4-way average.
// A slow-drifting low-frequency noise mask carves the frame into regions and
// each region shows ONE texture at full contrast (weights = area share), so
// bark reads as bark and ferns as ferns side by side. Sparkle is a separate
// additive layer: only its bright glints are added, its dark base discarded.
const FRAG_TEXSCENE = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform sampler2D uTex3;
uniform sampler2D uTexS;
uniform vec4 uW;
uniform float uSparkleW;
uniform sampler2D uFlow;
uniform sampler2D uDye;
uniform float uTime;
uniform float uSwirlPhase;
uniform float uWarp;
uniform float uTile;
uniform float uGlow;
uniform float uAspect;

vec2 mir (vec2 v) { return abs(fract(v * 0.5) * 2.0 - 1.0); } // seamless mirror-tile

float hash21 (vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise (vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
             mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm3 (vec2 p) {
  float v = 0.5 * vnoise(p);
  v += 0.25 * vnoise(p * 2.13 + 11.7);
  v += 0.125 * vnoise(p * 4.31 + 29.3);
  return v / 0.875;
}

void main () {
  vec2 p = vUv - 0.5;
  p.x *= uAspect;
  float r = length(p);
  float a = atan(p.y, p.x);
  // Spiral: inner content rotates faster than outer — hypnotic vortex pull.
  // uSwirlPhase is the CPU-integrated angle (never rate × absolute time).
  a += uSwirlPhase * (1.45 - r * 1.1);
  vec2 suv = vec2(cos(a), sin(a)) * r;
  suv.x /= uAspect;
  suv += 0.5;

  vec2 off = texture2D(uFlow, suv).xy;
  vec2 tuv = suv + off * uWarp;

  // Patch mask: sampled at the swirled (pre-warp) uv so regions ride the
  // spiral but hold their shape while content warps inside them; the slight
  // flow nudge lets borders billow with the fluid. Stretched around 0.5 so
  // cumulative-threshold area shares track the weights reasonably.
  float n = fbm3(suv * 2.6 + vec2(uTime * 0.008, -uTime * 0.005) + off * 0.35);
  n = clamp((n - 0.5) * 1.6 + 0.5, 0.0, 1.0);
  float t0 = uW.x;
  float t1 = uW.x + uW.y;
  float t2 = uW.x + uW.y + uW.z;
  float e = 0.055; // soft border half-width

  // De-phased tiling per slot so repeated textures don't lock step.
  vec3 c0 = texture2D(uTex0, mir(tuv * uTile)).rgb;
  vec3 c1 = texture2D(uTex1, mir(tuv * uTile * 1.17 + 0.31)).rgb;
  vec3 c2 = texture2D(uTex2, mir(tuv * uTile * 0.83 + 0.62)).rgb;
  vec3 c3 = texture2D(uTex3, mir(tuv * uTile * 1.31 + 0.13)).rgb;
  vec3 c = c0;
  c = mix(c, c1, smoothstep(t0 - e, t0 + e, n));
  c = mix(c, c2, smoothstep(t1 - e, t1 + e, n));
  c = mix(c, c3, smoothstep(t2 - e, t2 + e, n));

  // Additive sparkle glints (light-on-water reads as light, not as a dimmer).
  vec3 sp = texture2D(uTexS, mir(tuv * uTile * 1.08 + 0.47)).rgb;
  c += uSparkleW * max(sp - vec3(0.32), 0.0) * 2.4;

  vec3 dye = texture2D(uDye, suv).rgb;
  float glowLum = max(dye.r, max(dye.g, dye.b));
  // Texture is the substance; dye is bioluminescence. Only the dye's HOT spots
  // lift the texture (smoothstep gate) — otherwise accumulated dye floods the
  // whole frame and the dark ground the reference needs is lost.
  float g = smoothstep(0.2, 1.1, glowLum);
  // 0.85 baseline: the footage's own luminance detail (bark grain, ripple
  // edges) must survive to the display pass instead of sinking into the
  // black-lift crush; the dye still lifts hot spots above it.
  c = c * (0.85 + uGlow * g) + dye * 0.35 * g;
  gl_FragColor = vec4(c, 1.0);
}`;

const FRAG_KALEIDO = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float uOrderA;
uniform float uOrderB;
uniform float uOrderBlend;
uniform float uMix;
uniform float uRot;
uniform float uAspect;
uniform float uTime;
uniform float uMode; // 0 = composite (sparse dye), 1 = classic wedge fold (full-frame texture)

float khash21 (vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float knoise (vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(khash21(i), khash21(i + vec2(1.0, 0.0)), f.x),
             mix(khash21(i + vec2(0.0, 1.0)), khash21(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Classic kaleidoscope: fold the full frame into one mirrored wedge of
// 2*pi/order. Right for full-frame texture content (Visualiser 2.0) — every
// wedge is guaranteed to have content, and detail stays crisp. uRot spins
// which slice of the source is sampled (the mandala turns).
vec3 wedgeSample (float r, float baseA, float order) {
  float seg = 6.28318530718 / order;
  float a = mod(baseA + uRot, seg);
  a = abs(a - seg * 0.5);
  vec2 q = vec2(cos(a), sin(a)) * r;
  q.x /= uAspect;
  return texture2D(uTexture, clamp(q + 0.5, 0.0, 1.0)).rgb;
}

// Dihedral symmetrisation by compositing, not folding: max-blend all N
// rotated copies of the whole canvas plus their mirrors. Unlike a classic
// single-wedge kaleidoscope fold (which only samples one 2*pi/N slice and
// goes empty whenever the drifting dye masses happen to sit elsewhere), this
// uses every mass wherever it wanders, and the output is exactly N-fold
// mirror-symmetric by construction. uRot spins the copies (tunnel travel).
vec3 kalSample (float r, float baseA, float order) {
  vec3 acc = vec3(0.0);
  float seg = 6.28318530718 / order;
  for (int k = 0; k < 6; k++) {
    if (float(k) >= order) break;
    float a1 = baseA + seg * float(k) + uRot;
    float a2 = -baseA + seg * float(k) + uRot; // mirrored copy
    vec2 q1 = vec2(cos(a1), sin(a1)) * r;
    vec2 q2 = vec2(cos(a2), sin(a2)) * r;
    q1.x /= uAspect;
    q2.x /= uAspect;
    vec2 u1 = q1 + 0.5;
    vec2 u2 = q2 + 0.5;
    float ok1 = step(0.0, u1.x) * step(u1.x, 1.0) * step(0.0, u1.y) * step(u1.y, 1.0);
    float ok2 = step(0.0, u2.x) * step(u2.x, 1.0) * step(0.0, u2.y) * step(u2.y, 1.0);
    acc = max(acc, texture2D(uTexture, u1).rgb * ok1);
    acc = max(acc, texture2D(uTexture, u2).rgb * ok2);
  }
  return acc;
}

void main () {
  vec3 raw = texture2D(uTexture, vUv).rgb;
  if (uMix < 0.001) { gl_FragColor = vec4(raw, 1.0); return; }
  vec2 p = vUv - 0.5;
  p.x *= uAspect;
  float r = length(p);
  float baseA = atan(p.y, p.x);
  vec3 k;
  if (uMode > 2.5) {
    // Quadrant mirror ("moth"): reflect across both axes. Content stays
    // recognisable — this is a straight reflection, not a polar remap.
    vec2 q = vec2(0.5 - abs(vUv.x - 0.5), 0.5 - abs(vUv.y - 0.5));
    k = texture2D(uTexture, q).rgb;
  } else if (uMode > 1.5) {
    // Vertical axis mirror (Rorschach): left half reflected onto the right.
    k = texture2D(uTexture, vec2(0.5 - abs(vUv.x - 0.5), vUv.y)).rgb;
  } else if (uMode > 0.5) {
    vec3 kb = wedgeSample(r, baseA, uOrderB);
    k = kb;
    if (uOrderBlend < 0.999) {
      vec3 ka = wedgeSample(r, baseA, uOrderA);
      k = mix(ka, kb, uOrderBlend);
    }
  } else {
    vec3 kb = kalSample(r, baseA, uOrderB);
    k = kb;
    // Only pay for the outgoing order while a crossfade is actually running.
    if (uOrderBlend < 0.999) {
      vec3 ka = kalSample(r, baseA, uOrderA);
      k = mix(ka, kb, uOrderBlend);
    }
  }
  // Organic variation within the order: the fold amount varies subtly across
  // the frame (slow-drifting low-freq noise, ±12%), so parts of the mandala
  // resolve slightly out of perfect symmetry — living tissue, not clockwork.
  float nm = knoise(vUv * 3.1 + vec2(uTime * 0.013, -uTime * 0.009));
  float mixL = clamp(uMix * (1.0 + 0.24 * (nm - 0.5)), 0.0, 1.0);
  gl_FragColor = vec4(mix(raw, k, mixL), 1.0);
}`;

const FRAG_BLOOM_PREFILTER = `
precision mediump float;
precision mediump sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform vec3 curve;
uniform float threshold;
void main () {
  vec3 c = texture2D(uTexture, vUv).rgb;
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - curve.x, 0.0, curve.y);
  rq = curve.z * rq * rq;
  c *= max(rq, br - threshold) / max(br, 0.0001);
  gl_FragColor = vec4(c, 0.0);
}`;

const FRAG_BLOOM_BLUR = `
precision mediump float;
precision mediump sampler2D;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uTexture;
void main () {
  vec4 sum = vec4(0.0);
  sum += texture2D(uTexture, vL);
  sum += texture2D(uTexture, vR);
  sum += texture2D(uTexture, vT);
  sum += texture2D(uTexture, vB);
  sum *= 0.25;
  gl_FragColor = sum;
}`;

const FRAG_BLOOM_FINAL = `
precision mediump float;
precision mediump sampler2D;
varying vec2 vUv;
uniform sampler2D uTexture;
uniform float intensity;
void main () {
  vec4 sum = texture2D(uTexture, vUv);
  gl_FragColor = vec4(sum.rgb * intensity, 1.0);
}`;

const FRAG_SPLAT = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main () {
  vec2 p = vUv - point.xy;
  p.x *= aspectRatio;
  vec3 splat = exp(-dot(p, p) / radius) * color;
  vec3 base = texture2D(uTarget, vUv).xyz;
  gl_FragColor = vec4(base + splat, 1.0);
}`;

const FRAG_ADVECTION = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;

vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
  vec2 st = uv / tsize - 0.5;
  vec2 iuv = floor(st);
  vec2 fuv = fract(st);
  vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
  vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
  vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
  vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
  return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}

void main () {
#ifdef MANUAL_FILTERING
  vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
  vec4 result = bilerp(uSource, coord, dyeTexelSize);
#else
  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
  vec4 result = texture2D(uSource, coord);
#endif
  float decay = 1.0 + dissipation * dt;
  gl_FragColor = result / decay;
}`;

const FRAG_DIVERGENCE = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).x;
  float R = texture2D(uVelocity, vR).x;
  float T = texture2D(uVelocity, vT).y;
  float B = texture2D(uVelocity, vB).y;
  vec2 C = texture2D(uVelocity, vUv).xy;
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  if (vT.y > 1.0) { T = -C.y; }
  if (vB.y < 0.0) { B = -C.y; }
  float div = 0.5 * (R - L + T - B);
  gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}`;

const FRAG_CURL = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uVelocity, vL).y;
  float R = texture2D(uVelocity, vR).y;
  float T = texture2D(uVelocity, vT).x;
  float B = texture2D(uVelocity, vB).x;
  float vorticity = R - L - T + B;
  gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}`;

const FRAG_VORTICITY = `
precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
  float L = texture2D(uCurl, vL).x;
  float R = texture2D(uCurl, vR).x;
  float T = texture2D(uCurl, vT).x;
  float B = texture2D(uCurl, vB).x;
  float C = texture2D(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= curl * C;
  force.y *= -1.0;
  vec2 velocity = texture2D(uVelocity, vUv).xy;
  velocity += force * dt;
  velocity = min(max(velocity, -1000.0), 1000.0);
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}`;

const FRAG_PRESSURE = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
void main () {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  float divergence = texture2D(uDivergence, vUv).x;
  float pressure = (L + R + B + T - divergence) * 0.25;
  gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}`;

// Shared with textureLibrary.js (procedural texture generation runs in the
// same GL context, reusing this module's shader plumbing).
export { compileShader, Program, VERT_BASE };

const FRAG_GRADIENT_SUBTRACT = `
precision mediump float;
precision mediump sampler2D;
varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
void main () {
  float L = texture2D(uPressure, vL).x;
  float R = texture2D(uPressure, vR).x;
  float T = texture2D(uPressure, vT).x;
  float B = texture2D(uPressure, vB).x;
  vec2 velocity = texture2D(uVelocity, vUv).xy;
  velocity.xy -= vec2(R - L, T - B);
  gl_FragColor = vec4(velocity, 0.0, 1.0);
}`;
