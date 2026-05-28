const DEFAULT_VIEWER_OPTIONS = Object.freeze({
  canvasId: "3dgs-viewer",
  camera: Object.freeze({
    fov: 60,
    minDistance: 0.2,
    maxDistance: 10
  }),
  controls: Object.freeze({
    enableZoom: true,
    autoRotate: false
  }),
  model: Object.freeze({
    format: "splat",
    url: null
  })
});

function mergeNested(defaults, overrides) {
  return Object.fromEntries(
    Object.entries(defaults).map(([key, value]) => {
      if (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        overrides[key] &&
        typeof overrides[key] === "object"
      ) {
        return [key, { ...value, ...overrides[key] }];
      }

      return [key, overrides[key] ?? value];
    })
  );
}

function assertValidCamera(camera) {
  if (camera.fov <= 0 || camera.fov >= 180) {
    throw new RangeError("camera.fov must be greater than 0 and less than 180 degrees.");
  }

  if (camera.minDistance < 0 || camera.maxDistance <= camera.minDistance) {
    throw new RangeError("camera distances must satisfy 0 <= minDistance < maxDistance.");
  }
}

export function createViewerConfig(options = {}) {
  const merged = mergeNested(DEFAULT_VIEWER_OPTIONS, options);
  assertValidCamera(merged.camera);
  return merged;
}

export function resolveAssetUrl(baseUrl, assetPath) {
  if (!baseUrl || !assetPath) {
    throw new TypeError("baseUrl and assetPath are required.");
  }

  return new URL(assetPath, baseUrl).toString();
}

export function normalizePackageVersion(version) {
  const normalized = String(version).trim().replace(/^v/, "");
  if (!/^\d+\.\d+\.\d+$/.test(normalized)) {
    throw new TypeError("version must be a semantic version such as 1.0.1.");
  }

  return normalized;
}
