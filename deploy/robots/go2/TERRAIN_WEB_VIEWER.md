# Terrain decoder web viewer

`terrain_web_viewer` is a read-only DDS-to-HTTP bridge for the E2E policy's
reconstructed `17x11` terrain height map. It contains the complete web page and
has no Python, Node.js, OpenCV, or CDN dependency.

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

CMake selects the repository's x86_64 or aarch64 ONNX Runtime automatically,
so the same source builds on the simulation PC and Jetson Orin.

## x86 PC with MuJoCo

Start the MuJoCo depth publisher and `go2_ctrl`, enter `DepthVelocity`, then run:

```bash
./build/terrain_web_viewer --network lo --port 8080
```

Open <http://127.0.0.1:8080>. The viewer and `go2_ctrl` must use the same DDS
network interface; if the controller uses a different interface, replace `lo`
with that interface in the viewer command as well.

## Jetson Orin with the real robot

Run the controller and viewer with the same DDS network interface:

```bash
./build/go2_ctrl --network eth0
./build/terrain_web_viewer --network eth0 --port 8080
```

From another computer on the same network, open `http://JETSON_IP:8080`.

The viewer subscribes to `rt/terrain_decode`. Use `--topic TOPIC` to override
the topic and `/health` to check the HTTP service without opening a browser.
