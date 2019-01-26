# JAGS batchelder module

This module is not documented, and has not been fully reviewed. Not for distribution or use outside UCI MadLab.

contact icushman@uci.edu for details.

Tested working on Ubuntu 18.04.1 with JAGS 4.3.0 default install.

to install:
download this repo
go to repo directory in command line

enter:
`autoreconf -fvi`
`./configure`
`sudo make`
`sudo make install`

If your JAGS is properly installed in the default configuration, this should install `batchelder.so` and `batchelder.la` to your JAGS modules directory. On my system, this is `/usr/local/lib/x86_64-linux-gnu/JAGS/modules-4`.

If this works properly, you may then load the module, and access the Batchelder function using the JAGS code `theta = Batchelder(a, b, r, v, t, L1, L2)`.
