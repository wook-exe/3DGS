import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const packageJson = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
const source = await readFile(join(root, "src", "index.js"), "utf8");

await mkdir(join(root, "dist"), { recursive: true });
await writeFile(
  join(root, "dist", "index.js"),
  `// ${packageJson.name} v${packageJson.version}\n${source}`,
  "utf8"
);
