# Overview

Silly little app I made for my gf on Valentine's day. Pathtracing SDF renderer implemented in WGSL with some simple animation and phong illumination.

![](readme-assets/app.png)

# Running

If you happen to have an arm64-based macos system, you can run (obligatory disclaimer that you should always be careful running arbitrary binaries downloaded from the internet, run at your own risk, yada yada yada)
```bash
$ curl -L https://github.com/j-helland/sdf-valentine/releases/download/v0.0.0-macos-alpha/ilu.tar.gz | tar -xz; ./ilu
```

# Building From Source

The quickest way to build and run the app is
```bash
$ zig build run
```

To build the project for distribution, run
```bash
$ make release
```
which will generate a `release/ilu.tar.gz`.
