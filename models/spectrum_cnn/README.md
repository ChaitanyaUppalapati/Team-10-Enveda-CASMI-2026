# models/spectrum_cnn

**Owner:** Member 2. CNNs on binned spectra.

| Folder | Architecture | Schedule |
|--------|--------------|----------|
| [`arch1_cnn1d`](arch1_cnn1d/) | 1D CNN on binned spectra | week 2 |
| [`arch2_multires_cnn`](arch2_multires_cnn/) | Multi-resolution CNN | week 3 |

Shared code between the two architectures goes in `common/` inside this folder (create it
if needed). Don't import from other members' model folders; read their score files instead.
