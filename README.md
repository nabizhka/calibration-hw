# Two-period binomial calibration and model risk (MF731 HW 13, sheet `13_calibration_pub`)

sigma0 = lambda1/2. For lambda1 = .2, lambda2 = .4 the calibrated set is a broken line:
sigma1d = 1/3 for sigma1u in [0, 1/11]; 1.1 sigma1u + .9 sigma1d = .4 for sigma1u in [1/11, 3/11];
sigma1u = 3/11 for sigma1d in [0, 1/9]. Model risk of (S2 - 90)^+ with S0 = 100: 17.5 - 12.5 = 5.

```bash
pip install -r requirements.txt
python src/main.py        # asserts the checks, writes generated/ and figures/
latexmk -pdf main.tex
```
