// Atmospheric visual layer driven ONLY by the model's rain probability.
//
// These are illustrative visualisations of the forecast probability. They are
// not radar, satellite or observed precipitation, and no spatial rainfall field
// is drawn (the dataset has one value per district).
//
// Display thresholds (documented in the on-map legend):
//   tier 0  <10%    clear          no streaks
//   tier 1  10–29%  light          ~28 streaks, low opacity
//   tier 2  30–59%  moderate       ~70 streaks
//   tier 3  60–79%  heavy          ~120 streaks
//   tier 4  ≥80%    very heavy     ~170 streaks

export const TIERS = [
  { id: 0, min: 0, label: "Clear", streaks: 0, cloud: 0.10, alpha: 0 },
  { id: 1, min: 10, label: "Light", streaks: 28, cloud: 0.30, alpha: 0.28 },
  { id: 2, min: 30, label: "Moderate", streaks: 70, cloud: 0.46, alpha: 0.36 },
  { id: 3, min: 60, label: "Heavy", streaks: 120, cloud: 0.62, alpha: 0.44 },
  { id: 4, min: 80, label: "Very heavy", streaks: 170, cloud: 0.78, alpha: 0.52 },
];

export function tierFor(probabilityPercent) {
  if (probabilityPercent == null || Number.isNaN(probabilityPercent)) return null;
  let tier = TIERS[0];
  for (const t of TIERS) {
    if (probabilityPercent >= t.min) tier = t;
  }
  return tier;
}

export class WeatherFX {
  constructor({ frame, canvas, cloudField, tint, legendItems, reducedMotion }) {
    this.frame = frame;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cloudField = cloudField;
    this.tint = tint;
    this.legendItems = legendItems;
    this.reducedMotion = reducedMotion;

    this.targetStreaks = 0;
    this.currentStreaks = 0;
    this.streakAlpha = 0;
    this.targetAlpha = 0;
    this.particles = [];
    this.running = false;
    this.visible = true;
    this.lastTime = 0;
    this.rafId = null;
    this.width = 0;
    this.height = 0;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.activeTier = null;

    this.resize();
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(frame);
    this._tick = this._tick.bind(this);

    this.setProbability(null);
  }

  /** probability: percentage 0–100, or null for "no forecast" */
  setProbability(probability) {
    const tier = tierFor(probability);
    this.activeTier = tier;

    // Clouds: idle opacity when no forecast; otherwise tier-driven opacity.
    const cloud = tier ? tier.cloud : 0.10;
    document.documentElement.style.setProperty("--cloud-strength", String(cloud));

    // Subtle overall tint scaled by probability (static, works with reduced motion too).
    const tintValue = tier && probability != null ? Math.min(0.42, probability / 100 * 0.42) : 0;
    this.tint.style.setProperty("--rain-tint", tintValue.toFixed(3));

    this.targetStreaks = tier ? tier.streaks : 0;
    this.targetAlpha = tier ? tier.alpha : 0;

    this.legendItems.forEach((li) => {
      li.classList.toggle("is-active", !!tier && Number(li.dataset.tier) === tier.id);
    });

    if (!this.reducedMotion()) this._ensureRunning();
  }

  setVisible(isVisible) {
    this.visible = isVisible;
    this.cloudField.classList.toggle("is-paused", !isVisible);
    if (isVisible) this._ensureRunning();
  }

  resize() {
    const rect = this.frame.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  _ensureRunning() {
    if (this.running) return;
    const shouldRun =
      this.visible && !document.hidden && !this.reducedMotion() &&
      (this.currentStreaks > 0.5 || this.targetStreaks > 0 || this.streakAlpha > 0.01);
    if (!shouldRun) return;
    this.running = true;
    this.lastTime = performance.now();
    this.rafId = requestAnimationFrame(this._tick);
  }

  _stop() {
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  _spawn(fromTop) {
    return {
      x: Math.random() * (this.width + 40) - 20,
      y: fromTop ? -Math.random() * this.height : Math.random() * this.height,
      len: 10 + Math.random() * 12,
      speed: 380 + Math.random() * 260,
      alpha: 0.5 + Math.random() * 0.5,
    };
  }

  _tick(now) {
    if (!this.running) return;
    // Pause decorative animation when the map is off-screen, the tab is hidden,
    // or the user prefers reduced motion. _ensureRunning() restarts it later.
    if (!this.visible || document.hidden || this.reducedMotion()) {
      this._stop();
      return;
    }
    const dt = Math.min(0.05, (now - this.lastTime) / 1000);
    this.lastTime = now;

    // Ease the streak count and alpha toward the forecast tier (smooth transitions).
    this.currentStreaks += (this.targetStreaks - this.currentStreaks) * Math.min(1, dt * 2.2);
    this.streakAlpha += (this.targetAlpha - this.streakAlpha) * Math.min(1, dt * 2.2);

    const want = Math.round(this.currentStreaks);
    while (this.particles.length < want) this.particles.push(this._spawn(false));
    if (this.particles.length > want + 2) this.particles.length = want;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.lineWidth = 1;
    ctx.lineCap = "round";

    const slant = 0.22;
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      p.y += p.speed * dt;
      p.x += p.speed * slant * dt;
      if (p.y - p.len > this.height || p.x > this.width + 20) {
        Object.assign(p, this._spawn(true), { x: Math.random() * this.width });
      }
      ctx.strokeStyle = `rgba(190, 232, 255, ${(p.alpha * this.streakAlpha).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.len * slant, p.y - p.len);
      ctx.stroke();
    }

    const settled = this.targetStreaks === 0 && this.currentStreaks < 0.5 && this.streakAlpha < 0.01;
    if (settled) {
      this.particles.length = 0;
      this._stop();
      return;
    }
    this.rafId = requestAnimationFrame(this._tick);
  }
}
