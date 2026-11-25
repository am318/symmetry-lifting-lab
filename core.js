// Numerical core for the symmetry-lifting viewer. Plain JS, float64, no dependencies.
// Implements the NeurReps objects: P_E f (Reynolds average by quadrature), delta = f - P_E f,
// eta = dPi(xi) f, and the lifts c* = J^+ eta, b* = J^+(-delta) with the tau-truncated
// pseudoinverse in the sensitivity-normalised gauge.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- SiLU MLP
  const sig = (z) => 1 / (1 + Math.exp(-z));

  // Group-invariant input maps for equivariant-by-construction networks, with their forward-mode tangents.
  // "r2": (x^2 + y^2) [, alpha];  "c<n>": (r, cos n phi, sin n phi). Writes features into h and tangents into hd.
  function featureMap(kind) {
    if (kind === "r2") return (x, u, h, hd) => { h[0] = x[0] * x[0] + x[1] * x[1]; hd[0] = u ? 2 * (x[0] * u[0] + x[1] * u[1]) : 0; };
    const n = +kind.slice(1);
    return (x, u, h, hd) => {
      const r = Math.hypot(x[0], x[1]), p = Math.atan2(x[1], x[0]), c = r > 0 ? Math.cos(n * p) : 1, s = r > 0 ? Math.sin(n * p) : 0;
      h[0] = r; h[1] = c; h[2] = s;
      if (u && r > 0) { const dr = (x[0] * u[0] + x[1] * u[1]) / r, dp = (x[0] * u[1] - x[1] * u[0]) / (r * r); hd[0] = dr; hd[1] = -n * s * dp; hd[2] = n * c * dp; }
      else { hd[0] = 0; hd[1] = 0; hd[2] = 0; }
    };
  }

  function makeNet(spec) {
    // spec: [{W: [[...]] (out x in), b: [...]}] (an MLP on the raw inputs), or {feat, layers} for an MLP on
    // group-invariant inputs; flat params in PyTorch order (W then b, per layer)
    const layers = Array.isArray(spec) ? spec : spec.layers, feat = !Array.isArray(spec) && spec.feat ? featureMap(spec.feat) : null;
    const rd = !Array.isArray(spec) && spec.f32 ? Math.fround : (v) => v;   // float32 nets are stored in float32 digits
    const shapes = layers.map((l) => [l.W.length, l.W[0].length]);
    const P = shapes.reduce((s, [o, i]) => s + o * i + o, 0);
    const theta = new Float64Array(P);
    const offs = [];
    let k = 0;
    layers.forEach((l, li) => {
      const [o, i] = shapes[li];
      const wOff = k;
      for (let r = 0; r < o; r++) for (let c = 0; c < i; c++) theta[k++] = rd(l.W[r][c]);
      const bOff = k;
      for (let r = 0; r < o; r++) theta[k++] = rd(l.b[r]);
      offs.push({ wOff, bOff, o, i });
    });
    const width = Math.max(...shapes.map((s) => Math.max(s[0], s[1])));
    const L = offs.length;
    // scratch buffers
    const H = [], Hd = [], Z = [], Zd = [];
    for (let l = 0; l <= L; l++) { H.push(new Float64Array(width)); Hd.push(new Float64Array(width)); }
    for (let l = 0; l < L; l++) { Z.push(new Float64Array(width)); Zd.push(new Float64Array(width)); }
    const gh = new Float64Array(width), ghd = new Float64Array(width);
    const gz = new Float64Array(width), gzd = new Float64Array(width);

    // forward with one input tangent u; returns [v, vdot]
    function forward(th, x, u) {
      if (!u) return [value1(th, x), 0];
      if (feat) feat(x, u, H[0], Hd[0]);
      else { const nIn = offs[0].i; for (let j = 0; j < nIn; j++) { H[0][j] = x[j]; Hd[0][j] = u ? u[j] : 0; } }
      let v = 0, vd = 0;
      for (let l = 0; l < L; l++) {
        const { wOff, bOff, o, i } = offs[l];
        const h = H[l], hd = Hd[l];
        for (let r = 0; r < o; r++) {
          let z = th[bOff + r], zd = 0;
          const row = wOff + r * i;
          for (let c = 0; c < i; c++) { const w = th[row + c]; z += w * h[c]; zd += w * hd[c]; }
          if (l < L - 1) {
            Z[l][r] = z; Zd[l][r] = zd;
            const s = sig(z), sp = s * (1 + z * (1 - s));
            H[l + 1][r] = z * s; Hd[l + 1][r] = sp * zd;
          } else { v = z; vd = zd; }
        }
      }
      return [v, vd];
    }

    // value only (no tangent), still filling Z and H so backward() works after it
    function value1(th, x) {
      if (feat) feat(x, null, H[0], Hd[0]);
      else { const nIn = offs[0].i; for (let j = 0; j < nIn; j++) { H[0][j] = x[j]; Hd[0][j] = 0; } }
      let v = 0;
      for (let l = 0; l < L; l++) {
        const { wOff, bOff, o, i } = offs[l], h = H[l], last = l === L - 1;
        const hn = H[l + 1], hdn = Hd[l + 1], zl = Z[l], zdl = Zd[l];
        for (let r = 0; r < o; r++) {
          let z = th[bOff + r];
          const row = wOff + r * i;
          for (let c = 0; c < i; c++) z += th[row + c] * h[c];
          if (last) v = z;
          else { zl[r] = z; zdl[r] = 0; hn[r] = z / (1 + Math.exp(-z)); hdn[r] = 0; }
        }
      }
      return v;
    }

    // value and both input-direction tangents (u = e1, e2) in one pass; out = [v, dv/dx0, dv/dx1]
    const H1 = new Float64Array(width), H2 = new Float64Array(width), Ha = new Float64Array(width), Hb = new Float64Array(width);
    const G1 = new Float64Array(width), G2 = new Float64Array(width), Ga = new Float64Array(width), Gb = new Float64Array(width);
    const E1 = [1, 0, 0], E2 = [0, 1, 0];
    function grad2(th, x, out) {
      let h = Ha, g1 = G1, g2 = G2, hn = Hb, n1 = H1, n2 = H2;
      if (feat) { feat(x, E1, h, g1); feat(x, E2, fh, g2); }
      else { const nIn = offs[0].i; for (let j = 0; j < nIn; j++) { h[j] = x[j]; g1[j] = j === 0 ? 1 : 0; g2[j] = j === 1 ? 1 : 0; } }
      for (let l = 0; l < L; l++) {
        const { wOff, bOff, o, i } = offs[l], last = l === L - 1;
        for (let r = 0; r < o; r++) {
          let z = th[bOff + r], z1 = 0, z2 = 0;
          const row = wOff + r * i;
          for (let c = 0; c < i; c++) { const w = th[row + c]; z += w * h[c]; z1 += w * g1[c]; z2 += w * g2[c]; }
          if (last) { if (r === 0) { out[0] = z; out[1] = z1; out[2] = z2; } }
          else { const sg = 1 / (1 + Math.exp(-z)), sp = sg * (1 + z * (1 - sg)); hn[r] = z * sg; n1[r] = sp * z1; n2[r] = sp * z2; }
        }
        if (!last) {
          let t = h; h = hn; hn = t; t = g1; g1 = n1; n1 = t; t = g2; g2 = n2; n2 = t;
        }
      }
      return out;
    }

    // batched value: X holds N points with stride `st` (raw inputs first); writes N outputs
    let bufA = new Float64Array(0), bufB = new Float64Array(0);
    const fx = new Float64Array(4), fh = new Float64Array(width), fhd = new Float64Array(width);
    function values(th, X, st, N, out) {
      if (bufA.length < N * width) { bufA = new Float64Array(N * width); bufB = new Float64Array(N * width); }
      let A = bufA, B = bufB;
      const i0 = offs[0].i;
      if (feat) {
        for (let p = 0; p < N; p++) { for (let j = 0; j < st; j++) fx[j] = X[p * st + j]; feat(fx, null, fh, fhd); for (let c = 0; c < i0; c++) A[p * i0 + c] = fh[c]; }
      } else for (let p = 0; p < N; p++) for (let c = 0; c < i0; c++) A[p * i0 + c] = X[p * st + c];
      for (let l = 0; l < L; l++) {
        const { wOff, bOff, o, i } = offs[l], last = l === L - 1;
        let p = 0;
        // four points at a time: each weight is loaded once for four multiply-adds
        for (; p + 3 < N; p += 4) {
          const a0 = p * i, a1 = a0 + i, a2 = a1 + i, a3 = a2 + i;
          for (let r = 0; r < o; r++) {
            const bb = th[bOff + r], row = wOff + r * i;
            let z0 = bb, z1 = bb, z2 = bb, z3 = bb;
            for (let c = 0; c < i; c++) { const w = th[row + c]; z0 += w * A[a0 + c]; z1 += w * A[a1 + c]; z2 += w * A[a2 + c]; z3 += w * A[a3 + c]; }
            if (last) { if (r === 0) { out[p] = z0; out[p + 1] = z1; out[p + 2] = z2; out[p + 3] = z3; } }
            else {
              const b0 = p * o + r;
              B[b0] = z0 / (1 + Math.exp(-z0)); B[b0 + o] = z1 / (1 + Math.exp(-z1));
              B[b0 + 2 * o] = z2 / (1 + Math.exp(-z2)); B[b0 + 3 * o] = z3 / (1 + Math.exp(-z3));
            }
          }
        }
        for (; p < N; p++) {
          const a0 = p * i, b0 = p * o;
          for (let r = 0; r < o; r++) {
            let z = th[bOff + r];
            const row = wOff + r * i;
            for (let c = 0; c < i; c++) z += th[row + c] * A[a0 + c];
            if (last) { if (r === 0) out[p] = z; } else B[b0 + r] = z / (1 + Math.exp(-z));
          }
        }
        const t = A; A = B; B = t;
      }
      return out;
    }

    // gradient wrt params of (a * v + ad * vdot) at the last forward() point, accumulated into g
    function backward(th, a, ad, g) {
      let { wOff, bOff, o, i } = offs[L - 1];
      const hL = H[L - 1], hdL = Hd[L - 1];
      for (let c = 0; c < i; c++) {
        g[wOff + c] += a * hL[c] + ad * hdL[c];
        gh[c] = th[wOff + c] * a; ghd[c] = th[wOff + c] * ad;
      }
      g[bOff] += a;
      for (let l = L - 2; l >= 0; l--) {
        ({ wOff, bOff, o, i } = offs[l]);
        const z = Z[l], zd = Zd[l], h = H[l], hd = Hd[l];
        for (let r = 0; r < o; r++) {
          const s = sig(z[r]), sp = s * (1 + z[r] * (1 - s)), spp = s * (1 - s) * (2 + z[r] * (1 - 2 * s));
          gz[r] = gh[r] * sp + ghd[r] * spp * zd[r];
          gzd[r] = ghd[r] * sp;
        }
        for (let c = 0; c < i; c++) { gh[c] = 0; ghd[c] = 0; }
        for (let r = 0; r < o; r++) {
          const row = wOff + r * i, zr = gz[r], zdr = gzd[r];
          g[bOff + r] += zr;
          for (let c = 0; c < i; c++) {
            g[row + c] += zr * h[c] + zdr * hd[c];
            const w = th[row + c];
            gh[c] += w * zr; ghd[c] += w * zdr;
          }
        }
      }
    }
    return { theta, P, offs, forward, backward, values, grad2, shapes, feat: Array.isArray(spec) ? null : spec.feat || null };
  }

  // ---------------------------------------------------------------- systems
  // A system maps a 2D point to an output f(x) in R^d, with SO(2) acting on inputs and (for d = 2)
  // on outputs. It provides f and the parameter-Jacobian rows of f.
  function scalarSystem(net) {
    const x3 = new Float64Array(2);
    return {
      d: 1,
      field(th, x, y, out) { x3[0] = x; x3[1] = y; out[0] = net.forward(th, x3, null)[0]; },
      jacRows(th, x, y, rows) { x3[0] = x; x3[1] = y; net.forward(th, x3, null); net.backward(th, 1, 0, rows[0]); },
      value(th, x, y) { x3[0] = x; x3[1] = y; return net.forward(th, x3, null)[0]; },
      values(th, xy, N, out) { return net.values(th, xy, 2, N, out); },
    };
  }

  function forceSystem(net, getAlpha) {
    // f = -grad_q V(q; alpha); V_net input is (q1, q2, alpha)
    const inp = new Float64Array(3), e1 = [1, 0, 0], e2 = [0, 1, 0];
    let buf3 = new Float64Array(0); const g3 = new Float64Array(3);
    return {
      d: 2,
      field(th, x, y, out) {
        inp[0] = x; inp[1] = y; inp[2] = getAlpha();
        net.grad2(th, inp, g3); out[0] = -g3[1]; out[1] = -g3[2];
      },
      jacRows(th, x, y, rows) {
        inp[0] = x; inp[1] = y; inp[2] = getAlpha();
        net.forward(th, inp, e1); net.backward(th, 0, -1, rows[0]);
        net.forward(th, inp, e2); net.backward(th, 0, -1, rows[1]);
      },
      value(th, x, y) { inp[0] = x; inp[1] = y; inp[2] = getAlpha(); return net.forward(th, inp, null)[0]; },
      values(th, xy, N, out) {
        if (buf3.length < 3 * N) buf3 = new Float64Array(3 * N);
        const a = getAlpha();
        for (let p = 0; p < N; p++) { buf3[3 * p] = xy[2 * p]; buf3[3 * p + 1] = xy[2 * p + 1]; buf3[3 * p + 2] = a; }
        return net.values(th, buf3, 3, N, out);
      },
    };
  }

  function polarProbe(nR, nA, rMax) {
    const pts = [];
    for (let i = 1; i <= nR; i++) for (let j = 0; j < nA; j++) {
      const r = (rMax * i) / nR, a = (2 * Math.PI * j) / nA;
      pts.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    return pts;
  }

  // rho(g)^{-1} f(R_g x) averaged over nQuad rotations
  function reynoldsAt(sys, th, x, y, nQuad, out) {
    const d = sys.d, tmp = new Float64Array(d);
    out.fill(0);
    for (let k = 0; k < nQuad; k++) {
      const p = (2 * Math.PI * k) / nQuad, c = Math.cos(p), s = Math.sin(p);
      sys.field(th, c * x - s * y, s * x + c * y, tmp);
      if (d === 1) out[0] += tmp[0];
      else { out[0] += c * tmp[0] + s * tmp[1]; out[1] += -s * tmp[0] + c * tmp[1]; }
    }
    for (let j = 0; j < d; j++) out[j] /= nQuad;
  }

  // [Pi(g_phi) f](x) = rho(g) f(R_phi^{-1} x); eta = d/dphi at 0 by central difference
  function etaAt(sys, th, x, y, out) {
    const d = sys.d, h = 1e-5, a = new Float64Array(d), b = new Float64Array(d);
    const piG = (phi, o) => {
      const c = Math.cos(phi), s = Math.sin(phi);
      sys.field(th, c * x + s * y, -s * x + c * y, o);
      if (d === 2) { const u = o[0], v = o[1]; o[0] = c * u - s * v; o[1] = s * u + c * v; }
    };
    piG(h, a); piG(-h, b);
    for (let j = 0; j < d; j++) out[j] = (a[j] - b[j]) / (2 * h);
  }

  // f, P_E f, delta, eta stacked over probe points (length N*d)
  function probeState(sys, th, probe, nQuad, wantEta) {
    const d = sys.d, N = probe.length;
    const f = new Float64Array(N * d), pe = new Float64Array(N * d), eta = wantEta ? new Float64Array(N * d) : null;
    const t = new Float64Array(d);
    probe.forEach(([x, y], n) => {
      sys.field(th, x, y, t); for (let j = 0; j < d; j++) f[n * d + j] = t[j];
      reynoldsAt(sys, th, x, y, nQuad, t); for (let j = 0; j < d; j++) pe[n * d + j] = t[j];
      if (wantEta) { etaAt(sys, th, x, y, t); for (let j = 0; j < d; j++) eta[n * d + j] = t[j]; }
    });
    const delta = f.map((v, i) => v - pe[i]);
    return { f, pe, delta, eta, defect: norm(delta) / Math.max(norm(f), 1e-300) };
  }

  // Same f, P_E f, delta as probeState on a polar probe (nR radii x nA angles), but evaluating each ring
  // once at nQuad angles (nQuad a multiple of nA): the quadrature orbit of a ring point is the ring itself.
  function ringState(sys, th, nR, nA, nQuad, rMax, wantEta) {
    if (nQuad % nA) throw new Error("ringState: nQuad must be a multiple of nA");
    const d = sys.d, N = nR * nA, step = nQuad / nA;
    const f = new Float64Array(N * d), pe = new Float64Array(N * d), eta = wantEta ? new Float64Array(N * d) : null;
    const t = new Float64Array(d);
    for (let i = 0; i < nR; i++) {
      const r = (rMax * (i + 1)) / nR;
      let m0 = 0, m1 = 0; // ring mean in the local polar frame (radial, tangential) or scalar
      for (let k = 0; k < nQuad; k++) {
        const a = (2 * Math.PI * k) / nQuad, c = Math.cos(a), s = Math.sin(a);
        sys.field(th, r * c, r * s, t);
        if (d === 1) m0 += t[0]; else { m0 += c * t[0] + s * t[1]; m1 += -s * t[0] + c * t[1]; }
        if (k % step === 0) { const n = i * nA + k / step; for (let j = 0; j < d; j++) f[n * d + j] = t[j]; }
      }
      m0 /= nQuad; m1 /= nQuad;
      for (let j = 0; j < nA; j++) {
        const n = i * nA + j, a = (2 * Math.PI * j) / nA, c = Math.cos(a), s = Math.sin(a);
        if (d === 1) pe[n] = m0; else { pe[2 * n] = c * m0 - s * m1; pe[2 * n + 1] = s * m0 + c * m1; }
        if (wantEta) { etaAt(sys, th, r * c, r * s, t); for (let q = 0; q < d; q++) eta[n * d + q] = t[q]; }
      }
    }
    const delta = f.map((v, i) => v - pe[i]);
    return { f, pe, delta, eta, defect: norm(delta) / Math.max(norm(f), 1e-300) };
  }

  // f, P_G f, delta for the cyclic group C_n on a polar probe whose nA angles are a multiple of n:
  // every rotated probe point is again a probe point, so P_G f is a mean of probe values.
  function cyclicState(sys, th, nR, nA, n, rMax, wantEta) {
    if (nA % n) throw new Error("cyclicState: nA must be a multiple of n");
    const d = sys.d, N = nR * nA, sh = nA / n;
    const f = new Float64Array(N * d), pe = new Float64Array(N * d), eta = wantEta ? new Float64Array(N * d) : null;
    const t = new Float64Array(d);
    for (let i = 0; i < nR; i++) for (let j = 0; j < nA; j++) {
      const r = (rMax * (i + 1)) / nR, a = (2 * Math.PI * j) / nA, x = r * Math.cos(a), y = r * Math.sin(a), q = i * nA + j;
      sys.field(th, x, y, t); for (let k = 0; k < d; k++) f[q * d + k] = t[k];
      if (wantEta) { etaAt(sys, th, x, y, t); for (let k = 0; k < d; k++) eta[q * d + k] = t[k]; }
    }
    for (let i = 0; i < nR; i++) for (let j = 0; j < nA; j++) {
      const q = i * nA + j;
      for (let k = 0; k < n; k++) {
        const qq = i * nA + ((j + k * sh) % nA);
        if (d === 1) pe[q] += f[qq];
        else { // rho(g)^{-1} f(R_g x), g = rotation by 2 pi k / n
          const p = (2 * Math.PI * k) / n, c = Math.cos(p), s = Math.sin(p);
          pe[2 * q] += c * f[2 * qq] + s * f[2 * qq + 1]; pe[2 * q + 1] += -s * f[2 * qq] + c * f[2 * qq + 1];
        }
      }
      for (let k = 0; k < d; k++) pe[q * d + k] /= n;
    }
    const delta = f.map((v, i) => v - pe[i]);
    return { f, pe, delta, eta, defect: norm(delta) / Math.max(norm(f), 1e-300) };
  }

  const norm = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  // ---------------------------------------------------------------- Jacobian and lift
  function jacobian(sys, th, P, probe) {
    const d = sys.d, rows = [];
    const tmp = [];
    for (let j = 0; j < d; j++) tmp.push(null);
    for (const [x, y] of probe) {
      for (let j = 0; j < d; j++) tmp[j] = new Float64Array(P);
      sys.jacRows(th, x, y, tmp);
      for (let j = 0; j < d; j++) rows.push(tmp[j]);
    }
    return rows; // array of M rows, each Float64Array(P)
  }

  // Symmetric eigendecomposition (Householder tridiagonalisation + implicit QL), in place.
  // a: array of Float64Array rows (n x n). Returns eigenvalues d; a becomes eigenvectors (columns).
  function symEig(a) {
    const n = a.length, d = new Float64Array(n), e = new Float64Array(n);
    for (let i = n - 1; i > 0; i--) {
      const l = i - 1; let h = 0, scale = 0;
      if (l > 0) {
        for (let k = 0; k <= l; k++) scale += Math.abs(a[i][k]);
        if (scale === 0) e[i] = a[i][l];
        else {
          for (let k = 0; k <= l; k++) { a[i][k] /= scale; h += a[i][k] * a[i][k]; }
          let f = a[i][l]; let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
          e[i] = scale * g; h -= f * g; a[i][l] = f - g; f = 0;
          // p = A u from the lower triangle, row by row (contiguous access): row j gives sum_k<=j a_jk u_k to p_j
          // and a_jk u_j to p_k for k < j
          const u = a[i];
          for (let j = 0; j <= l; j++) e[j] = 0;
          for (let j = 0; j <= l; j++) {
            a[j][i] = u[j] / h;
            const row = a[j], uj = u[j]; let sj = 0;
            for (let k = 0; k < j; k++) { const v = row[k]; sj += v * u[k]; e[k] += v * uj; }
            e[j] += sj + row[j] * uj;
          }
          for (let j = 0; j <= l; j++) { e[j] /= h; f += e[j] * u[j]; }
          g = 0;
          const hh = f / (h + h);
          for (let j = 0; j <= l; j++) {
            f = a[i][j]; e[j] = g = e[j] - hh * f;
            for (let k = 0; k <= j; k++) a[j][k] -= f * e[k] + g * a[i][k];
          }
        }
      } else e[i] = a[i][l];
      d[i] = h;
    }
    d[0] = 0; e[0] = 0;
    const g = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const l = i - 1;
      if (d[i]) { // row-wise form of the accumulation (same arithmetic, contiguous access)
        const ai = a[i];
        g.fill(0, 0, l + 1);
        for (let k = 0; k <= l; k++) { const w = ai[k], ak = a[k]; if (w) for (let j = 0; j <= l; j++) g[j] += w * ak[j]; }
        for (let k = 0; k <= l; k++) { const w = a[k][i], ak = a[k]; if (w) for (let j = 0; j <= l; j++) ak[j] -= g[j] * w; }
      }
      d[i] = a[i][i]; a[i][i] = 1;
      for (let j = 0; j <= l; j++) a[j][i] = a[i][j] = 0;
    }
    // QL on the transposed vectors: Z[j] is column j of a, so each rotation touches two contiguous rows
    const Z = [];
    for (let j = 0; j < n; j++) { const z = new Float64Array(n); for (let k = 0; k < n; k++) z[k] = a[k][j]; Z.push(z); }
    for (let i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    for (let l = 0; l < n; l++) {
      let iter = 0, m;
      do {
        for (m = l; m < n - 1; m++) {
          const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) <= 1e-15 * dd) break;
        }
        if (m !== l) {
          if (iter++ === 60) break;
          let g = (d[l + 1] - d[l]) / (2 * e[l]);
          let r = Math.hypot(g, 1);
          g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
          let s = 1, c = 1, p = 0, i;
          for (i = m - 1; i >= l; i--) {
            let f = s * e[i]; const b = c * e[i];
            e[i + 1] = r = Math.hypot(f, g);
            if (r === 0) { d[i + 1] -= p; e[m] = 0; break; }
            s = f / r; c = g / r; g = d[i + 1] - p; r = (d[i] - g) * s + 2 * c * b;
            d[i + 1] = g + (p = s * r); g = c * r - b;
            const z0 = Z[i], z1 = Z[i + 1];
            for (let k = 0; k < n; k++) { f = z1[k]; z1[k] = s * z0[k] + c * f; z0[k] = c * z0[k] - s * f; }
          }
          if (r === 0 && i >= l) continue;
          d[l] -= p; e[l] = g; e[m] = 0;
        }
      } while (m !== l);
    }
    for (let j = 0; j < n; j++) { const z = Z[j]; for (let k = 0; k < n; k++) a[k][j] = z[k]; }
    a.vecs = Z; // eigenvectors as rows, for callers that want them that way
    return d;
  }

  // Top eigenpairs of a symmetric PSD matrix by orthogonalised subspace iteration; grows the block until
  // the tau cutoff falls inside it. Returns {lam, U} with U an array of eigenvector columns (Float64Array M).
  // When the resolved rank is a sizeable fraction of M, a full eigendecomposition is faster than growing
  // the block, so that path is taken directly (and remembered for the next solve of the same size).
  const bigRank = {};
  function fullEig(K) {
    const A = K.map((r) => Float64Array.from(r)), lam = symEig(A);
    return { lam, U: A.vecs };
  }
  function topEig(K, tau, k0) {
    const M = K.length;
    if (bigRank[M]) {
      const res = fullEig(K); let lmax = 0; for (const l of res.lam) lmax = Math.max(lmax, l);
      bigRank[M] = Array.from(res.lam).filter((l) => l > 0 && Math.sqrt(l) >= tau * Math.sqrt(lmax)).length > M / 6;
      return res;
    }
    let k = Math.min(M, k0 || 64);
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
    const orth = (Q) => {
      for (let pass = 0; pass < 2; pass++) for (let a = 0; a < Q.length; a++) {
        for (let b = 0; b < a; b++) { const p = dot(Q[a], Q[b]); const qa = Q[a], qb = Q[b]; for (let i = 0; i < M; i++) qa[i] -= p * qb[i]; }
        const nn = norm(Q[a]) || 1; for (let i = 0; i < M; i++) Q[a][i] /= nn;
      }
      return Q;
    };
    const mulK = (Q) => Q.map((q) => { const y = new Float64Array(M); for (let m = 0; m < M; m++) y[m] = dot(K[m], q); return y; });
    for (;;) {
      let Q = [];
      for (let a = 0; a < k; a++) { const q = new Float64Array(M); for (let i = 0; i < M; i++) q[i] = rnd(); Q.push(q); }
      Q = orth(mulK(Q));
      for (let it = 0; it < 6; it++) Q = orth(mulK(Q));
      const KQ = mulK(Q);
      const B = Q.map((qa) => Float64Array.from(KQ, (kq) => dot(qa, kq)));
      for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) { const v = 0.5 * (B[a][b] + B[b][a]); B[a][b] = v; B[b][a] = v; }
      const lam = symEig(B); // B -> eigenvectors of the small problem
      let lmax = 0; for (const l of lam) lmax = Math.max(lmax, l);
      const nkeep = Array.from(lam).filter((l) => l > 0 && Math.sqrt(l) >= tau * Math.sqrt(lmax)).length;
      if (nkeep < k - 12 || k >= M) {
        const U = [];
        for (let j = 0; j < k; j++) { const u = new Float64Array(M); for (let a = 0; a < k; a++) { const w = B[a][j]; if (w) { const q = Q[a]; for (let i = 0; i < M; i++) u[i] += w * q[i]; } } U.push(u); }
        return { lam, U };
      }
      k = Math.min(M, 2 * k);
      if (k > M / 3) { bigRank[M] = nkeep > M / 6; return fullEig(K); }
    }
  }

  // Eigenpairs of K = J J^T without forming K: pivoted Cholesky K ~ L L^T, stopped once every remaining
  // diagonal entry is below 1e-13 of the largest (far under the tau^2 = 1e-6 eigenvalue cutoff), then the small
  // eigenproblem of L^T L. Only the pivot columns of K are computed, so the cost scales with the numerical rank.
  function gram(J) { // K = J J^T, four rows at a time so each row of J is read once per block
    const M = J.length, P = J[0].length, K = [];
    for (let m = 0; m < M; m++) K.push(new Float64Array(M));
    // 4 x 4 register blocks: each pass over the columns reads 8 rows of J for 16 entries of K
    const C = 512;   // column chunk, so the 8 row segments stay in cache
    const row = (m) => J[Math.min(m, M - 1)];
    for (let m = 0; m < M; m += 4) {
      const a0 = row(m), a1 = row(m + 1), a2 = row(m + 2), a3 = row(m + 3);
      for (let n = 0; n <= Math.min(m + 3, M - 1); n += 4) {
        const b0 = row(n), b1 = row(n + 1), b2 = row(n + 2), b3 = row(n + 3);
        const S = new Float64Array(16);
        for (let c0 = 0; c0 < P; c0 += C) {
          const c1 = Math.min(P, c0 + C);
          let s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0, s20 = 0, s21 = 0, s22 = 0, s23 = 0, s30 = 0, s31 = 0, s32 = 0, s33 = 0;
          for (let i = c0; i < c1; i++) {
            const x0 = a0[i], x1 = a1[i], x2 = a2[i], x3 = a3[i], y0 = b0[i], y1 = b1[i], y2 = b2[i], y3 = b3[i];
            s00 += x0 * y0; s01 += x0 * y1; s02 += x0 * y2; s03 += x0 * y3;
            s10 += x1 * y0; s11 += x1 * y1; s12 += x1 * y2; s13 += x1 * y3;
            s20 += x2 * y0; s21 += x2 * y1; s22 += x2 * y2; s23 += x2 * y3;
            s30 += x3 * y0; s31 += x3 * y1; s32 += x3 * y2; s33 += x3 * y3;
          }
          S[0] += s00; S[1] += s01; S[2] += s02; S[3] += s03; S[4] += s10; S[5] += s11; S[6] += s12; S[7] += s13;
          S[8] += s20; S[9] += s21; S[10] += s22; S[11] += s23; S[12] += s30; S[13] += s31; S[14] += s32; S[15] += s33;
        }
        for (let t = 0; t < 4 && m + t < M; t++) for (let u = 0; u < 4 && n + u < M; u++) if (n + u <= m + t) { K[m + t][n + u] = S[4 * t + u]; K[n + u][m + t] = S[4 * t + u]; }
      }
    }
    return K;
  }
  // sizes M whose last solve was nearly full rank: go straight to the dense path rather than spend M/3 pivoted
  // Cholesky steps first only to abandon them (the result is identical either way, only the cost differs)
  const denseM = {};
  function denseEig(J, tol) {
    const res = fullEig(gram(J));
    let q = 0; for (const l of res.lam) if (l > tol) q++;
    denseM[J.length] = q > J.length / 4;
    return res;
  }
  function gramEig(J) {
    const M = J.length, d = new Float64Array(M);
    for (let m = 0; m < M; m++) d[m] = dot(J[m], J[m]);
    let d0 = 0; for (const v of d) d0 = Math.max(d0, v);
    const L = [], tol = 1e-13 * d0;
    if (denseM[M]) return denseEig(J, tol);
    while (L.length < M) {
      if (L.length > M / 3) return denseEig(J, tol); // nearly full rank: forming K and a dense solve is cheaper
      let p = 0; for (let m = 1; m < M; m++) if (d[m] > d[p]) p = m;
      if (!(d[p] > tol)) break;
      const jp = J[p], col = new Float64Array(M), P = jp.length;
      let m = 0;
      for (; m + 3 < M; m += 4) {   // four rows per pass: each entry of jp is loaded once for four products
        const r0 = J[m], r1 = J[m + 1], r2 = J[m + 2], r3 = J[m + 3];
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let i = 0; i < P; i++) { const v = jp[i]; s0 += r0[i] * v; s1 += r1[i] * v; s2 += r2[i] * v; s3 += r3[i] * v; }
        col[m] = s0; col[m + 1] = s1; col[m + 2] = s2; col[m + 3] = s3;
      }
      for (; m < M; m++) col[m] = dot(J[m], jp);
      for (const l of L) { const lp = l[p]; if (lp) for (let m = 0; m < M; m++) col[m] -= lp * l[m]; }
      const piv = Math.sqrt(d[p]);
      for (let m = 0; m < M; m++) { col[m] /= piv; d[m] -= col[m] * col[m]; }
      d[p] = 0; L.push(col);
    }
    const q = L.length, B = [];
    for (let a = 0; a < q; a++) B.push(new Float64Array(q));
    for (let a = 0; a < q; a++) for (let b = 0; b <= a; b++) { const v = dot(L[a], L[b]); B[a][b] = v; B[b][a] = v; }
    const lam = symEig(B), U = [];
    for (let j = 0; j < q; j++) {
      const u = new Float64Array(M), sj = Math.sqrt(Math.max(lam[j], 1e-300));
      for (let a = 0; a < q; a++) { const w = B[a][j] / sj; if (w) { const l = L[a]; for (let m = 0; m < M; m++) u[m] += w * l[m]; } }
      U.push(u);
    }
    return { lam, U };
  }

  // Min-norm, tau-truncated lift in the sensitivity-normalised gauge, as in the paper:
  // truncate the SVD of the RAW J at sigma_k >= tau * sigma_1, then among c with Vh_k c = S_k^-1 U_k^T y
  // minimise ||diag(||J_:,i||) c|| (sensitivity-normalised gauge); dead columns get exactly 0.
  function lift(J, targets, tau, raw, mask) {
    const M = J.length, P = J[0].length;
    const col = new Float64Array(P);
    for (const r of J) for (let i = 0; i < P; i++) col[i] += r[i] * r[i];
    let cmax = 0;
    for (let i = 0; i < P; i++) { col[i] = Math.sqrt(col[i]); cmax = Math.max(cmax, col[i]); }
    if (mask) { // only the masked parameters are free: zero the other columns before solving
      const fullCol = col;
      J = J.map((r) => { const z = new Float64Array(P); for (let i = 0; i < P; i++) if (mask[i]) z[i] = r[i]; return z; });
      const res = lift(J, targets, tau, raw, null); res[res.length - 1].colnorm = fullCol; return res;
    }
    const dead = new Uint8Array(P), cs = new Float64Array(P);
    for (let i = 0; i < P; i++) { dead[i] = !raw && col[i] <= cmax * 1e-8 ? 1 : 0; cs[i] = raw ? 1 : Math.max(col[i], cmax * 1e-8); }
    const { lam, U } = gramEig(J);
    let lmax = 0;
    for (let k = 0; k < lam.length; k++) lmax = Math.max(lmax, lam[k]);
    const keep = [];
    for (let k = 0; k < lam.length; k++) if (lam[k] > 0 && Math.sqrt(lam[k]) >= tau * Math.sqrt(lmax)) keep.push(k);
    const r = keep.length;
    // Vh_k rows: v_k = J^T u_k / sigma_k ; a_eq = Vh_k / cs (dead -> 0)
    const aeq = keep.map((k) => {
      const v = new Float64Array(P), sk = Math.sqrt(lam[k]);
      for (let m = 0; m < M; m++) { const u = U[k][m] / sk; if (u === 0) continue; const row = J[m]; for (let i = 0; i < P; i++) v[i] += row[i] * u; }
      for (let i = 0; i < P; i++) v[i] = dead[i] ? 0 : v[i] / cs[i];
      return v;
    });
    // Gram G = a_eq a_eq^T (r x r), Cholesky
    const G = [];
    for (let a = 0; a < r; a++) { G.push(new Float64Array(r)); for (let b = 0; b <= a; b++) G[a][b] = dot(aeq[a], aeq[b]); }
    for (let a = 0; a < r; a++) {
      for (let b = 0; b <= a; b++) {
        let s = G[a][b];
        for (let k = 0; k < b; k++) s -= G[a][k] * G[b][k];
        G[a][b] = a === b ? Math.sqrt(Math.max(s, 1e-300)) : s / G[b][b];
      }
    }
    const solve = (z) => {
      const w = Float64Array.from(z);
      for (let a = 0; a < r; a++) { for (let k = 0; k < a; k++) w[a] -= G[a][k] * w[k]; w[a] /= G[a][a]; }
      for (let a = r - 1; a >= 0; a--) { for (let k = a + 1; k < r; k++) w[a] -= G[k][a] * w[k]; w[a] /= G[a][a]; }
      return w;
    };
    return targets.map((y) => {
      const coords = keep.map((k) => dot(U[k], y));
      const alpha = coords.map((c, j) => c / Math.sqrt(lam[keep[j]]));
      const w = solve(alpha);
      const c = new Float64Array(P);
      for (let a = 0; a < r; a++) { const wa = w[a], v = aeq[a]; for (let i = 0; i < P; i++) c[i] += v[i] * wa; }
      for (let i = 0; i < P; i++) c[i] /= cs[i];
      const ny = norm(y);
      let proj2 = 0; for (const cc of coords) proj2 += cc * cc;
      return { coef: c, residual: Math.sqrt(Math.max(ny * ny - proj2, 0)) / Math.max(ny, 1e-300), rank: r };
    }).concat([{ colnorm: col, rank: r }]);
  }

  root.SymCore = { gram, cyclicState, ringState, topEig, gramEig, makeNet, scalarSystem, forceSystem, polarProbe, reynoldsAt, etaAt, probeState, jacobian, lift, symEig, norm, dot };
  if (typeof module !== "undefined") module.exports = root.SymCore;
})(typeof window !== "undefined" ? window : globalThis);
