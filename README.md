# wimp_spectrum
Result
![FIG](figures/WIMP_upperlimit.png)

---

## Repository structure

```

wimp_spectrum/
├── WIMP_N_el_spectra/                output: elastic recoil spectrum files (generated)
├── figures/                          output figures
├── WIMP_xsec_LZ_excludedtupperlimit.txt   LZ upper limit on WIMP-nucleon cross-section
├── scatter_kinematics.py             recoil energy kinematics (Er_max)
├── scatter_target.py                 target nucleus properties (A, Z, number density)
├── scatter_xsec.py                   differential cross-section dσ/dEr with Helm form factor
├── DM_halo.py                        SHM velocity distribution, v_min, g(v_min)
├── DM_modulation_model.py            annual modulation, Standard Halo model
├── compute_el_spectrum.py            compute dN/dEr [ton⁻¹ yr⁻¹ keV⁻¹] and save to disk
├── get_upperlimit_spectrum.py        load spectra, rescale to σ_f, load exclusion curves
├── check_gvmin.py                    diagnostic checks on g(v_min)
├── plot_wimp_spectrum.py             plotting routines
└── plot_setup.py                     matplotlib style configuration
```

---

## Related repositories

This repository expects the following sibling directories:

```
~/projects/
├── neutrino_spectrum/      ← https://github.com/YZHUANGwork/neutrino_spectrum 
├── wimp_spectrum/          ← this repo
└── detector_efficiency/    ← https://github.com/YZHUANGwork/detector_efficiency
└── phase-split/           ← https://github.com/YZHUANGwork/phase-split
```

Detector efficiency files are read from `../detector_efficiency/` by default.
Clone all three repositories into the same parent folder.

---

computing elastic WIMP–nucleus nuclear recoil spectra.

```bash
python compute_el_spectrum.py
```
target Xenon, sigma_0 * halo

Outputs one file per WIMP mass to `WIMP_N_el_spectra/`, e.g. `WIMP-Nel_50GeV-Xe_1e-45cm2_pdf.txt`.


scale sigma_0 to current LZ exclusion limits
Standard halo model Ad, different masses
```python
from get_upperlimit_spectrum import load_exclusion_curve, get_exclusion_xsec

masses, xsecs, interp = load_exclusion_curve("WIMP_xsec_LZ_excludedtupperlimit.txt")
sigma_excl = get_exclusion_xsec(50 * u.GeV, interp)
Er, dN_dEr = get_WIMP_pdf(50 * u.GeV, sigma_excl, target="Xe")
```
