# @wook-exe/3dgs-viewer

Reusable JavaScript utilities for the 3DGS-Viewer web client.

## Install

```bash
npm install @wook-exe/3dgs-viewer --registry=https://npm.pkg.github.com
```

## Usage

```js
import { createViewerConfig, resolveAssetUrl } from "@wook-exe/3dgs-viewer";

const config = createViewerConfig({
  model: { url: resolveAssetUrl("/assets/", "product.splat") },
  controls: { autoRotate: true }
});
```

## Version

Current assignment release: `1.0.1`.
