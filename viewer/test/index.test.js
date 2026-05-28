import assert from "node:assert/strict";
import test from "node:test";
import {
  createViewerConfig,
  normalizePackageVersion,
  resolveAssetUrl
} from "../src/index.js";

test("createViewerConfig merges nested viewer options", () => {
  const config = createViewerConfig({
    camera: { fov: 75 },
    controls: { autoRotate: true },
    model: { url: "https://example.com/model.splat" }
  });

  assert.equal(config.camera.fov, 75);
  assert.equal(config.camera.minDistance, 0.2);
  assert.equal(config.controls.autoRotate, true);
  assert.equal(config.model.format, "splat");
});

test("createViewerConfig rejects invalid camera values", () => {
  assert.throws(() => createViewerConfig({ camera: { fov: 180 } }), RangeError);
  assert.throws(
    () => createViewerConfig({ camera: { minDistance: 2, maxDistance: 1 } }),
    RangeError
  );
});

test("resolveAssetUrl resolves relative model paths", () => {
  assert.equal(
    resolveAssetUrl("https://cdn.example.com/assets/", "product.splat"),
    "https://cdn.example.com/assets/product.splat"
  );
});

test("normalizePackageVersion supports the 1.0.0 to 1.0.1 release path", () => {
  assert.equal(normalizePackageVersion("v1.0.1"), "1.0.1");
  assert.throws(() => normalizePackageVersion("1.0"), TypeError);
});
