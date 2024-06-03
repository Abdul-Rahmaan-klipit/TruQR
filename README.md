# TruQR
- Python = 3.9
- The following Flask app detected QR codes and sends images for decoding at the server side.
- The detected frames undergo preprocessing for detection of inner QR via edge detection.
- Both the outer as well as inner QR are passed for similarity check
- Reprogrammed Working API for passing entire frame as input and performing backend operations
- - Error formed:  Max retries exceeded with url
  - Resolvation attempts: creating session, try-exception, passing additional paramters (proxies, verify=False), status: unresolved
