// Precompute bge-small document embeddings for the site's semantic search.
// Usage: node embed.mjs <texts.json> <out embeddings.bin>
// Needs @huggingface/transformers (same major version the page loads) resolvable from cwd or NODE_PATH.
import { readFileSync, writeFileSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";

const [, , inPath, outPath] = process.argv;
const texts = JSON.parse(readFileSync(inPath, "utf8"));
const DIM = 384, BATCH = 32;
const embed = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" });
const out = new Int8Array(texts.length * DIM);
for (let i = 0; i < texts.length; i += BATCH) {
  const batch = texts.slice(i, i + BATCH);
  const t = await embed(batch, { pooling: "cls", normalize: true });
  const data = t.data;
  for (let j = 0; j < batch.length; j++)
    for (let k = 0; k < DIM; k++)
      out[(i + j) * DIM + k] = Math.max(-127, Math.min(127, Math.round(data[j * DIM + k] * 127)));
  if ((i / BATCH) % 10 === 0) process.stderr.write(`${Math.min(i + BATCH, texts.length)}/${texts.length}\n`);
}
writeFileSync(outPath, Buffer.from(out.buffer));
console.log(`wrote ${texts.length} x ${DIM} int8 embeddings to ${outPath}`);
