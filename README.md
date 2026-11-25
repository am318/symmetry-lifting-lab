# Symmetry Lifting Lab

An interactive, in-browser companion to *Parameter-Level Attribution of Symmetry in Trained Networks* (NeurReps 2026).

**Live page:** https://am318.github.io/symmetry-lifting-lab/

Pick a trained network, then move its weights and watch the symmetry of what it computes:

- **Along c\*** = J⁺η: the parameter direction that rotates the learned function along its symmetry orbit.
- **Along b\*** = J⁺(−δ): the direction that shrinks the equivariance defect δ = f − P_ℰ f. **Restore symmetry** walks along b\* until the network computes the symmetrised function P_ℰ f.
- **Parameter groups:** select the top-k parameters by |b\*|, |c\*| or sensitivity S (or the biases), see how much of each direction the group carries, kick or zero it, or walk with only the group free.

Systems: an annulus classifier (trained on a 90° wedge or on all angles), the learned force of a Mexican-hat Hamiltonian network, a six-fold potential (trained on a 120° wedge or on the whole disc, with C₂, C₃, C₆ or SO(2) as the target group), a three-arm spiral classifier (C₃), and a two-body potential under softened gravity learned from three half orbits or 24 full orbits. Each network can be loaded at initialisation, after 2% and 10% of training, or fully trained, and a table compares every network's training loss with its symmetry defect.

Everything is computed live in the browser in float64: the forward pass, the Jacobian, the truncated pseudo-inverse and the walks. Nothing is sent anywhere.

## Files

| File | What it is |
|---|---|
| `index.html` | the page |
| `core.js` | networks, Jacobians, Reynolds averages and the lift solver |
| `worker.js` | runs the lift off the main thread |
| `weights.json` | the trained networks' weights |

## Running locally

Any static file server works, for example:

```bash
python3 -m http.server 8000
```

then open http://localhost:8000. Opening `index.html` directly from disk will not work, because browsers block `fetch` and web workers on `file://` pages.

## Licence

No licence is granted. Copyright © 2026 am318, all rights reserved. You may view and use the page on this site; please ask before copying, modifying or redistributing the code or weights.
