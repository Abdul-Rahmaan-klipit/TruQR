# TruQR

<div align="center">
	<img src="https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/resources/comparison/outer_with_inner_qr.png">
</div>

### Recommended to use [Python 3.9](https://www.python.org/downloads/release/python-390/)

### Steps to run
- Create Environment
```
conda create -n qr_env python==3.9
conda activate qr_env
```

- Install [Dependencies](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/requirements.txt)
```
pip install -r requirements.txt
```

- Run [app.py](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/api.py) for server initialization.
```
python app.py
```
- Run [main.py](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/main.py) for real-time detection and authentication.
```
python main.py
```

### Generating TruQR
- For Generating QR Code run [Generate-QR-Final.ipynb](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/Generate-QR-Final.ipynb)

### Decoding TruQR
- For decoding [Authentic Captured Image](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/main/resources/captured/Captured_TruQR.png), do the following steps:
  - Resize to [79x79](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/abc7e4f99ce857084e4e3e73c762a19dd62b1c63/Generate-QR-Final.ipynb#L666)
  - Convert to binary with a threshold of [90](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/abc7e4f99ce857084e4e3e73c762a19dd62b1c63/Generate-QR-Final.ipynb#L653).
  - Crop out the inner QR of size 23x23.
  - [Decode](https://github.com/Abdul-Rahmaan-klipit/TruQR/blob/abc7e4f99ce857084e4e3e73c762a19dd62b1c63/Generate-QR-Final.ipynb#L503) the QR using the valid seed.

### Contributions
> [Abdul Rahmaan](https://github.com/Abdul-Rahmaan-klipit)

> [Megha Manoj](https://github.com/mochi-bunny)
