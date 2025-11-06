# pickle5 compatibility shim

This package re-exports Python's built-in `pickle` module so that projects
which still depend on the historical `pickle5` backport can run on modern
Python versions (3.8+) where protocol 5 support is already included.
