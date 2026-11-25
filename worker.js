// Lift solver off the main thread: c* = J^+ eta and b* = J^+(-delta) at the posted parameters.
// m.group: 0 = SO(2) (Reynolds average by quadrature), n > 0 = cyclic group C_n.
// m.mask (optional Uint8Array): only those parameters are free; the solve uses only their columns of J.
importScripts("core.js");
const C = self.SymCore;
const nets = {};

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.cmd === "init") {
    for (const [k, layers] of Object.entries(m.weights)) nets[k] = C.makeNet(layers);
    return;
  }
  if (m.cmd === "lift") {
    const net = nets[m.key];
    const sys = m.kind === "scalar" ? C.scalarSystem(net) : C.forceSystem(net, () => m.alpha);
    const [nR, nA] = m.probe;
    const th = new Float64Array(m.theta);
    const st = m.group > 0 ? C.cyclicState(sys, th, nR, nA, m.group, 1, true)
      : C.ringState(sys, th, nR, nA, Math.max(64, 2 * nA), 1, true);
    const J = C.jacobian(sys, th, net.P, C.polarProbe(nR, nA, 1));
    const mask = m.mask ? new Uint8Array(m.mask) : null;
    const [c, b, info] = C.lift(J, [st.eta, st.delta.map((v) => -v)], 1e-3, !!m.raw, mask);
    const sens = info.colnorm.map((v) => v / Math.sqrt(J.length));
    self.postMessage({ id: m.id, c: c.coef, b: b.coef, resC: c.residual, resB: b.residual, rank: info.rank, sens, defect: st.defect },
      [c.coef.buffer, b.coef.buffer, sens.buffer]);
  }
};
