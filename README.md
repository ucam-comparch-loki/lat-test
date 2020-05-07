# Loki Accelerator Template test suite

Verify that the output computed by the Loki Accelerator Template is correct.

Each test is intended to be relatively quick and simple, with the complexity coming from running the same tests for many different accelerator configurations.

## Prerequisites

Compiling this code requires:
* The [Loki compiler](no_link_yet)
* [libloki](https://github.com/ucam-comparch-loki/libloki) for access to various Loki-specific features
* The [driver](https://github.com/ucam-comparch-loki/lat-ifc) for the Loki accelerator template
* The [neural network library](https://github.com/ucam-comparch-loki/lat-nn) for the Loki accelerator template

## Build

```
export LIBLOKI_DIR=path/to/libloki
export LAT_IFC_DIR=path/to/lat-ifc
export LAT_NN_DIR=path/to/lat-nn
make
```

## Usage

Running this code requires [lokisim](https://github.com/ucam-comparch-loki/lokisim/tree/accelerator) (accelerator branch).

To run all tests (requires the `lokisim` executable to be accessible from your `PATH`):
```
make test
```
Information will be displayed as testing progresses.

To run all tests for a single accelerator configuration:
```
lokisim [accelerator configuration] build/lat-test"
```
The return code is used to indicate the test result:
 * `0`: test passed
 * `-1`: program failure outside of a test (e.g. out of memory, invalid arguments)
 * anything else: ID of the first failing test

This all assumes that the simulated program exits normally. If the simulator itself crashes (which is a distinct possibility during testing), its return code overrides that of the simulated program. Refer to stdout and stderr to determine the source of any errors.

To run a single test:
```
lokisim [accelerator configuration] build/lat-test --test=ID
```
