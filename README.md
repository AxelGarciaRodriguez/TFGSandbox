# TFGSandbox_2nd

PyKinectV2 changes:
1. ```pip install pykinect2```
2. Remove line 2216 in Lib/site-packages/pykinect2/PyKinectV2.py (needs to be adjusted to work on a 64 bit System)
```python
# assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
```
3. Add the following lines below:
```python
import numpy.distutils.system_info as sysinfo
required_size = 64 + sysinfo.platform_bits / 4
assert sizeof(tagSTATSTG) == required_size, sizeof(tagSTATSTG)
```
4. Add version comptypes to line 2867:
```python
from comtypes import _check_version; _check_version('1.4.5')
```
