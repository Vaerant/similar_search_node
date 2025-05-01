const fs = require('fs/promises');
const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const cliProgress = require('cli-progress');

let embedder;

// Load environment variables from .env file
dotenv.config();

// Define sermon data structure (for TypeScript, this would be interfaces)
// But we'll use JSDoc annotations for better IDE support

/**
 * @typedef {Object} Section
 * @property {string} Paragraph - Paragraph identifier
 * @property {Array<Array<string>>} Content - Content blocks
 */

/**
 * @typedef {Object} Sermon
 * @property {string} id - Sermon ID
 * @property {string} title - Sermon title
 * @property {string} date - Sermon date
 * @property {Array<Section>} sections - Sermon sections
 * @property {string} _file - Source filename
 */

/**
 * @typedef {Object} TextMetadata
 * @property {number} sermon_idx - Index of the sermon
 * @property {string} sermon_id - ID of the sermon
 * @property {string} sermon_title - Title of the sermon
 * @property {string} sermon_date - Date of the sermon
 * @property {string} paragraph_id - Paragraph identifier
 * @property {number} block_idx - Index of the block within paragraph
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} sermon_title - Title of the sermon
 * @property {string} sermon_date - Date of the sermon
 * @property {string} sermon_id - ID of the sermon
 * @property {string} paragraph - Paragraph identifier
 * @property {number} content_index - Index of the content block
 * @property {string} text - Text content
 * @property {number} score - Search score
 */

// Parse command line arguments
const args = yargs(hideBin(process.argv))
  .option('query', {
    alias: 'q',
    type: 'string',
    description: 'Search query'
  })
  .option('top-k', {
    alias: 'k',
    type: 'number',
    default: 5,
    description: 'Number of results to return'
  })
  .option('rebuild', {
    alias: 'r',
    type: 'boolean',
    default: false,
    description: 'Force rebuild index'
  })
  .option('debug', {
    alias: 'd',
    type: 'boolean',
    default: false,
    description: 'Enable debug output'
  })
  .option('data-file', {
    alias: 'f',
    type: 'string',
    default: 'all_sermons.json',
    description: 'Path to sermon data JSON file'
  })
  .help()
  .argv;

// File paths
const SERMON_FILE = args.dataFile;
const EMBEDDINGS_CACHE_FILE = 'sermon_embeddings.json';
const METADATA_CACHE_FILE = 'sermon_metadata.json';
const TEXTS_CACHE_FILE = 'sermon_texts.json';

// Constants
const EMBEDDING_DIM = 1536; // OpenAI embedding dimension

/**
 * Initialize the local embedder
 */
async function initEmbedder() {
  // dynamically load the ES module in a CommonJS context
  const { pipeline } = await import('@xenova/transformers');
  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}

/**
 * Create embeddings locally (mean‑pool + normalize)
 * @param {Array<string>} texts
 * @returns {Promise<Array<Array<number>>>}
 */
async function createLocalEmbeddings(texts) {
  // show progress
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(texts.length, 0);

  const all = [];
  for (let i = 0; i < texts.length; i++) {
    const feats = await embedder(texts[i], { pooling: 'mean', normalize: true });
    all.push(Array.from(feats.data));
    progressBar.increment();
  }

  progressBar.stop();
  return all;
}

/**
 * Main application function
 */
async function main() {
  try {
    console.log("Sermon Search Application");
    
    // Initialize local embedder before any embedding calls
    console.log("Initializing local embedder…");
    await initEmbedder();

    // Check if we need to rebuild or load from cache
    const rebuildNeeded = args.rebuild || 
      !await fileExists(METADATA_CACHE_FILE) || 
      !await fileExists(TEXTS_CACHE_FILE) ||
      !await fileExists(EMBEDDINGS_CACHE_FILE);
      
    if (args.debug) {
      console.log("Debug mode enabled");
      console.log("Using local embedder for embeddings");
      console.log("Cache files exist check:");
      console.log(`  metadata_cache: ${await fileExists(METADATA_CACHE_FILE)}`);
      console.log(`  texts_cache: ${await fileExists(TEXTS_CACHE_FILE)}`);
      console.log(`  embeddings_cache: ${await fileExists(EMBEDDINGS_CACHE_FILE)}`);
      console.log(`Rebuild needed: ${rebuildNeeded}`);
    }
    
    let texts, metadata, embeddings;
    
    if (!rebuildNeeded) {
      console.log("Loading cached data...");
      
      // replace the unicode hyphen “utf‑8” with ASCII “utf8”
      texts    = JSON.parse(await fs.readFile(TEXTS_CACHE_FILE, 'utf8'));
      metadata = JSON.parse(await fs.readFile(METADATA_CACHE_FILE, 'utf8'));
      embeddings = JSON.parse(await fs.readFile(EMBEDDINGS_CACHE_FILE, 'utf8'))
        .map(v => Array.isArray(v) ? v : Object.values(v));

      console.log(`Loaded ${texts.length} cached text blocks with embeddings`);
    } else {
      console.log("Processing sermon data...");
      
      // Load sermons from file
      const sermons = await loadSermonData(SERMON_FILE);
      console.log(`Processed ${sermons.length} sermons from ${SERMON_FILE}`);
      
      // Extract text and metadata from sermons
      const { processedTexts, processedMetadata } = flattenSermonContent(sermons);
      texts = processedTexts;
      metadata = processedMetadata;
      console.log(`Indexed ${texts.length} text blocks from ${sermons.length} sermons`);
      
      // Use local embeddings instead of OpenAI
      console.log("Creating embeddings locally using Xenova pipeline…");
      const startTime = Date.now();

      // Open the cache file and write a JSON array bracket
      const fd = await fs.open(EMBEDDINGS_CACHE_FILE, 'w');
      await fd.write('[');

      const batchSize = 1000;
      for (let i = 0; i < texts.length; i += batchSize) {
        const slice = texts.slice(i, i + batchSize);
        const batchEmbs = await createLocalEmbeddings(slice);

        // write each embedding with commas
        for (let j = 0; j < batchEmbs.length; j++) {
          const isFirst = (i + j === 0);
          const prefix = isFirst ? '' : ',';
          await fd.write(prefix + JSON.stringify(batchEmbs[j]));
        }
      }

      // close JSON array and file
      await fd.write(']');
      await fd.close();

      console.log(`Created and cached embeddings in ${(Date.now() - startTime)/1000}s`);

      // load into memory if you still need them for searching
      embeddings = JSON.parse(await fs.readFile(EMBEDDINGS_CACHE_FILE, 'utf8'));
      
      console.log("Saving data to cache...");
      await fs.writeFile(TEXTS_CACHE_FILE, JSON.stringify(texts));
      await fs.writeFile(METADATA_CACHE_FILE, JSON.stringify(metadata));
    }
    
    // Instead of FAISS, just treat embeddings array as “index”
    const index = embeddings;

    // Search if query is provided
    if (args.query) {
      console.log(`Searching for query: '${args.query}'`);
      await searchAndDisplayResults(index, args.query, args.topK, texts, metadata);
    } else {
      console.log("No search query provided. Use --query to search.");
    }
    
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

/**
 * Load sermon data from JSON file
 * @param {string} filePath - Path to the JSON file
 * @returns {Promise<Array<Sermon>>} Array of sermon objects
 */
async function loadSermonData(filePath) {
  try {
    // change 'utf-8' -> 'utf8'
    const data = await fs.readFile(filePath, 'utf8');
    const sermons = JSON.parse(data);
    
    // Add filename to each sermon
    return sermons.map(sermon => {
      sermon._file = path.basename(filePath);
      return sermon;
    });
  } catch (error) {
    throw new Error(`Failed to load sermon data: ${error.message}`);
  }
}

/**
 * Extract text blocks and metadata from sermons
 * @param {Array<Sermon>} sermons - Array of sermon objects
 * @returns {Object} Object containing processed texts and metadata
 */
function flattenSermonContent(sermons) {
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(sermons.length, 0);
  
  const processedTexts = [];
  const processedMetadata = [];
  
  for (let sermonIdx = 0; sermonIdx < sermons.length; sermonIdx++) {
    const sermon = sermons[sermonIdx];
    
    for (const section of sermon.sections) {
      const paraId = section.Paragraph;
      
      for (let blockIdx = 0; blockIdx < section.Content.length; blockIdx++) {
        const block = section.Content[blockIdx];
        
        // Filter out empty strings
        const filteredBlock = block
          .map(s => s.trim())
          .filter(s => s.length > 0);
        
        // Skip if block is empty
        if (filteredBlock.length === 0) continue;
        
        // Join the text items
        const text = filteredBlock.join(' ');
        if (text.length > 10) {
          processedMetadata.push({
            sermon_idx: sermonIdx,
            sermon_id: sermon.id,
            sermon_title: sermon.title,
            sermon_date: sermon.date,
            paragraph_id: paraId,
            block_idx: blockIdx,
          });
          processedTexts.push(text);
        }
      }
    }
    progressBar.update(sermonIdx + 1);
  }
  
  progressBar.stop();
  return { processedTexts, processedMetadata };
}

/**
 * Brute‑force cosine similarity search
 */
async function searchAndDisplayResults(embeddings, query, topK, texts, metadata) {
  // Replace OpenAI query embedding with local
  const [qvec] = await createLocalEmbeddings([query]);

  // compute cosine sim = dot(a,b)/(|a||b|)
  const norms = embeddings.map(v => Math.hypot(...v));
  const qnorm = Math.hypot(...qvec);
  const scores = embeddings.map((v,i) => {
    const dot = v.reduce((sum, x, j) => sum + x * qvec[j], 0);
    return { idx: i, sim: dot / (norms[i] * qnorm) };
  });

  // pick topK
  scores
    .sort((a,b) => b.sim - a.sim)
    .slice(0, topK)
    .forEach((hit, rank) => {
      const meta = metadata[hit.idx];
      console.log(`${rank+1}. [${meta.sermon_title}] (${meta.sermon_date}) | ¶${meta.paragraph_id} blk ${meta.block_idx}`);
      console.log(`   Score: ${hit.sim.toFixed(4)}`);
      console.log(`   Text: ${texts[hit.idx]}\n`);
    });
}

/**
 * Search sermons and return an array of results.
 * @param {string} query
 * @param {Object} [options]
 * @param {number} [options.topK=5]        Number of results to return (0 for no limit)
 * @param {boolean}[options.rebuild=false] Force rebuild index
 * @param {string} [options.dataFile=all_sermons.json]
 * @param {boolean}[options.debug=false]
 * @param {number} [options.minScore=0.0]  Minimum cosine‐similarity to include
 * @returns {Promise<Array<SearchResult>>}
 */
async function searchSermons(query, {
  topK     = args.topK,
  rebuild  = args.rebuild,
  dataFile = args.dataFile,
  debug    = args.debug,
  minScore = 0.0
} = {}) {
  // override CLI args if running as module
  args.query    = query;
  args.topK     = topK;
  args.rebuild  = rebuild;
  args.dataFile = dataFile;
  args.debug    = debug;
  args.minScore = minScore;

  // init embedder
  await initEmbedder();

  // decide if we need to rebuild or load from cache
  const rebuildNeeded = rebuild ||
    !await fileExists(METADATA_CACHE_FILE) ||
    !await fileExists(TEXTS_CACHE_FILE) ||
    !await fileExists(EMBEDDINGS_CACHE_FILE);

  let texts, metadata, embeddings;

  if (!rebuildNeeded) {
    texts      = JSON.parse(await fs.readFile(TEXTS_CACHE_FILE, 'utf8'));
    metadata   = JSON.parse(await fs.readFile(METADATA_CACHE_FILE, 'utf8'));
    embeddings = JSON.parse(await fs.readFile(EMBEDDINGS_CACHE_FILE, 'utf8'))
                   .map(v => Array.isArray(v) ? v : Object.values(v));
  } else {
    const sermons = await loadSermonData(dataFile);
    const flat    = flattenSermonContent(sermons);
    texts    = flat.processedTexts;
    metadata = flat.processedMetadata;

    // rebuild embeddings cache
    const fd = await fs.open(EMBEDDINGS_CACHE_FILE, 'w');
    await fd.write('[');
    for (let i = 0; i < texts.length; i += 1000) {
      const slice = texts.slice(i, i + 1000);
      const batch = await createLocalEmbeddings(slice);
      for (let j = 0; j < batch.length; j++) {
        const prefix = (i + j === 0) ? '' : ',';
        await fd.write(prefix + JSON.stringify(batch[j]));
      }
    }
    await fd.write(']');
    await fd.close();

    embeddings = JSON.parse(await fs.readFile(EMBEDDINGS_CACHE_FILE, 'utf8'));
    await fs.writeFile(TEXTS_CACHE_FILE,    JSON.stringify(texts));
    await fs.writeFile(METADATA_CACHE_FILE, JSON.stringify(metadata));
  }

  // perform search & return structured results
  return await _searchAndReturnResults(embeddings, query, topK, texts, metadata, minScore);
}

async function _searchAndReturnResults(embeddings, query, topK, texts, metadata, minScore) {
  const [qvec] = await createLocalEmbeddings([query]);
  const norms  = embeddings.map(v => Math.hypot(...v));
  const qnorm  = Math.hypot(...qvec);

  // compute and sort by similarity
  const sorted = embeddings
    .map((v, i) => {
      const dot = v.reduce((sum, x, j) => sum + x * qvec[j], 0);
      return { idx: i, sim: dot / (norms[i] * qnorm) };
    })
    .sort((a, b) => b.sim - a.sim);

  // filter by minScore then apply topK
  const filtered = sorted.filter(hit => hit.sim >= minScore);
  const hits     = topK > 0 ? filtered.slice(0, topK) : filtered;

  return hits.map((hit, rank) => {
    const m = metadata[hit.idx];
    return {
      rank:          rank + 1,
      sermon_title:  m.sermon_title,
      sermon_date:   m.sermon_date,
      sermon_id:     m.sermon_id,
      paragraph:     m.paragraph_id,
      content_index: m.block_idx,
      text:          texts[hit.idx],
      score:         hit.sim
    };
  });
}

// keep CLI behavior when run directly
if (require.main === module) {
  main();
}

module.exports = { searchSermons };