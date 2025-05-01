const fs = require('fs/promises');
const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const cliProgress = require('cli-progress');
const { HierarchicalNSW } = require('hnswlib-node');
const ort = require('onnxruntime-node');
// ort.env = ort.LogLevel.WARNING;

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
    default: 'all_sermons2.json',
    description: 'Path to sermon data JSON file'
  })
  .option('limit', {
    alias: 'l',
    type: 'number',
    description: 'Number of sermons to process (default: process all)'
  })
  .option('direction', {
    alias: 'dir',
    type: 'string',
    choices: ['start', 'end', 'both'],
    default: 'both',
    description: 'Process sermons from start, end, or both directions'
  })
  .help()
  .argv;

// File paths
const SERMON_FILE = args.dataFile;
const EMBEDDINGS_CACHE_FILE = 'sermon_embeddings.json';
const METADATA_CACHE_FILE = 'sermon_metadata.json';
const TEXTS_CACHE_FILE = 'sermon_texts.json';
const INDEX_CACHE_FILE = 'sermon_index.faiss';

// Constants
const EMBEDDING_DIM = 384; // Updated embedding dimension

// New global for our local embedder
let embedder;

/**
 * Initialize the local embedder
 */
async function initEmbedder() {
  try {

    const { pipeline, env } = await import('@xenova/transformers');
    let ort;
    try {
      ort = require('onnxruntime-node');
    } catch (err) {
      console.warn('Failed to load onnxruntime-node binding:', err.message);
      throw err;
    }
    env.onnx = ort;

    // Use CUDA on Linux if available
    const device = process.platform === 'linux' ? 'cuda' : 
                  (process.platform === 'win32' ? 'dml' : 'cpu');

    try {
      // embedder = await pipeline('feature-extraction', 'Xenova/bge-large-en-v1.5', {
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        device,
        revision: 'main',
        quantized: false,
      });
      console.log(`Using ${device.toUpperCase()} for embedding`);
    } catch (err) {
      console.warn(`Failed to init ${device}, falling back to CPU:`, err.message);
      // embedder = await pipeline('feature-extraction', 'Xenova/bge-large-en-v1.5', {
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        device: 'cpu',
        revision: 'main',
        quantized: false,
      });
      console.log('Using CPU for embedding');
    }
  } catch (err) {
    console.error('Failed to initialize embedder:', err.message);
    throw err;
  }
}

/**
 * Normalize a vector to unit length (L2 norm)
 * @param {Array<number>} vector 
 * @returns {Array<number>}
 */
function normalizeVector(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return vector.map(val => val / magnitude);
}

/**
 * Create embeddings locally via Xenova with progress indication
 * @param {Array<string>} texts 
 * @returns {Promise<Array<Array<number>>>}
 */
async function createEmbeddingsLocal(texts) {
  if (!embedder) {
    await initEmbedder();
  }
  
  // Create progress bar
  const progressBar = new cliProgress.SingleBar({
    format: 'Creating embeddings [{bar}] {percentage}% | {value}/{total} texts | ETA: {eta}s',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
  }, cliProgress.Presets.shades_classic);
  
  progressBar.start(texts.length, 0);

  const all = [];
  for (let i = 0; i < texts.length; i++) {
    const feats = await embedder(texts[i], { pooling: 'mean', normalize: true });
    
    // Handle different possible output formats
    if (feats.data) {
      // If it's a tensor-like object with data property
      all.push(Array.from(feats.data));
    } else if (Array.isArray(feats)) {
      // If it's already an array (possibly nested)
      all.push(Array.isArray(feats[0]) ? feats[0] : feats);
    } else {
      throw new Error("Unexpected embedding output format");
    }
    
    progressBar.increment();
  }

  progressBar.stop();
  
  // Ensure all vectors are of correct dimension
  const validEmbeddings = all.map(vec => {
    if (vec.length !== EMBEDDING_DIM) {
      console.warn(`Warning: Found embedding with incorrect dimension: ${vec.length}, expected ${EMBEDDING_DIM}`);
      // Pad or truncate to correct dimension
      return vec.length > EMBEDDING_DIM ? 
        vec.slice(0, EMBEDDING_DIM) : 
        [...vec, ...Array(EMBEDDING_DIM - vec.length).fill(0)];
    }
    return vec;
  });
  
  // Normalize all vectors
  return validEmbeddings.map(normalizeVector);
}

/**
 * Main application function
 */
async function main() {
  try {
    console.log("Sermon Search Application");
    
    // Check if we need to rebuild or load from cache
    const rebuildNeeded = args.rebuild || 
      !await fileExists(METADATA_CACHE_FILE) || 
      !await fileExists(TEXTS_CACHE_FILE) ||
      !await fileExists(EMBEDDINGS_CACHE_FILE);
      
    if (args.debug) {
      console.log("Debug mode enabled");
      console.log("Using local embeddings via Xenova");
      console.log("Cache files exist check:");
      console.log(`  metadata_cache: ${await fileExists(METADATA_CACHE_FILE)}`);
      console.log(`  texts_cache: ${await fileExists(TEXTS_CACHE_FILE)}`);
      console.log(`  embeddings_cache: ${await fileExists(EMBEDDINGS_CACHE_FILE)}`);
      console.log(`Rebuild needed: ${rebuildNeeded}`);
    }
    
    let texts, metadata, embeddings;
    
    if (!rebuildNeeded) {
      console.log("Loading cached data...");
      
      // Load texts, metadata, and embeddings from cache files
      texts = JSON.parse(await fs.readFile(TEXTS_CACHE_FILE, 'utf-8'));
      metadata = JSON.parse(await fs.readFile(METADATA_CACHE_FILE, 'utf-8'));
      embeddings = JSON.parse(await fs.readFile(EMBEDDINGS_CACHE_FILE, 'utf-8'));
      
      console.log(`Loaded ${texts.length} cached text blocks with embeddings`);
    } else {
      console.log("Processing sermon data...");
      
      // Load sermons from file with filtering options
      const sermons = await loadSermonData(SERMON_FILE, {
        limit: args.limit,
        direction: args.direction
      });
      console.log(`Processed ${sermons.length} sermons from ${SERMON_FILE}`);
      
      // The rest of the function remains the same
      // Extract text and metadata from sermons
      const { processedTexts, processedMetadata } = flattenSermonContent(sermons);
      texts = processedTexts;
      metadata = processedMetadata;
      console.log(`Indexed ${texts.length} text blocks from ${sermons.length} sermons`);
      
      // Create embeddings using Xenova pipeline
      console.log("Creating embeddings locally using Xenova pipeline…");
      const startTime = Date.now();
      embeddings = await createEmbeddingsLocal(texts);
      const timeTaken = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`Created ${embeddings.length} embeddings in ${timeTaken}s (${(embeddings.length / parseFloat(timeTaken)).toFixed(2)} embeddings/sec)`);
      
      // Save data to cache files
      console.log("Saving data to cache...");
      await fs.writeFile(TEXTS_CACHE_FILE, JSON.stringify(texts));
      await fs.writeFile(METADATA_CACHE_FILE, JSON.stringify(metadata));
      await fs.writeFile(EMBEDDINGS_CACHE_FILE, JSON.stringify(embeddings));
    }

    // Build an ANN index using hnswlib-node
    console.log("Building ANN index using hnswlib-node...");

    // 1) Filter embeddings, texts & metadata to only correct‐sized vectors
    const validIndices = [];
    embeddings.forEach((vec, i) => {
      if (Array.isArray(vec) && vec.length === EMBEDDING_DIM) {
        validIndices.push(i);
      } else {
        console.warn(`Skipping embedding #${i}: expected length ${EMBEDDING_DIM}, got ${vec.length}`);
      }
    });
    const filteredEmbeddings = validIndices.map(i => embeddings[i]);
    const filteredTexts      = validIndices.map(i => texts[i]);
    const filteredMetadata   = validIndices.map(i => metadata[i]);

    // 2) Build index over filteredEmbeddings
    const annIndex = new HierarchicalNSW('cosine', EMBEDDING_DIM);
    annIndex.initIndex(filteredEmbeddings.length);
    filteredEmbeddings.forEach((vec, idx) => {
      annIndex.addPoint(vec, idx);
    });

    // 3) Wire filteredTexts/filteredMetadata into search call
    if (args.query) {
      console.log(`Searching for query: '${args.query}'`);
      await searchAndDisplayResults(
        annIndex,
        args.query,
        args.topK,
        filteredTexts,
        filteredMetadata
      );
    } else {
      console.log("No search query provided. Use --query to search.");
    }
    
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

/**
 * Load sermon data from JSON file with filtering options
 * @param {string} filePath - Path to the JSON file
 * @param {Object} options - Filtering options
 * @param {number} options.limit - Number of sermons to process (optional)
 * @param {string} options.direction - Direction for processing ('start', 'end', 'both')
 * @returns {Promise<Array<Sermon>>} Array of sermon objects
 */
async function loadSermonData(filePath, options = {}) {
  try {
    const data = await fs.readFile(filePath, 'utf-8');
    const allSermons = JSON.parse(data);
    
    let sermons = allSermons;
    
    // Apply filtering based on limit and direction
    if (options.limit) {
      const limit = Math.min(options.limit, allSermons.length);
      
      if (options.direction === 'start') {
        // Take from start
        sermons = allSermons.slice(0, limit);
        console.log(`Processing ${limit} sermons from the beginning`);
      } 
      else if (options.direction === 'end') {
        // Take from end
        sermons = allSermons.slice(-limit);
        console.log(`Processing ${limit} sermons from the end`);
      }
      else if (options.direction === 'both') {
        // Take half from start, half from end
        const halfLimit = Math.ceil(limit / 2);
        const fromStart = allSermons.slice(0, halfLimit);
        const fromEnd = allSermons.slice(-Math.floor(limit / 2));
        sermons = [...fromStart, ...fromEnd];
        console.log(`Processing ${fromStart.length} sermons from start and ${fromEnd.length} from end`);
      }
    } else {
      console.log(`Processing all ${allSermons.length} sermons`);
    }
    
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
      
      // Process paragraph as a whole for better context
      if (section.Content.length > 0) {
        // Combine all content in the paragraph for better context
        const allText = section.Content
          .flat()
          .map(s => s.trim())
          .filter(s => s.length > 0)
          .join(' ');
          
        if (allText.length > 30) {  // Only include if there's enough content
          processedMetadata.push({
            sermon_idx: sermonIdx,
            sermon_id: sermon.id,
            sermon_title: sermon.title,
            sermon_date: sermon.date,
            paragraph_id: paraId,
            block_idx: -1, // -1 indicates full paragraph
          });
          processedTexts.push(cleanText(allText));
        }
      }
      
      // Also process individual blocks for more granular matches
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
        if (text.length > 30) {  // Increased minimum length for better context
          processedMetadata.push({
            sermon_idx: sermonIdx,
            sermon_id: sermon.id,
            sermon_title: sermon.title,
            sermon_date: sermon.date,
            paragraph_id: paraId,
            block_idx: blockIdx,
          });
          processedTexts.push(cleanText(text));
        }
      }
    }
    progressBar.update(sermonIdx + 1);
  }
  
  progressBar.stop();
  return { processedTexts, processedMetadata };
}

/**
 * Clean and normalize text for better embedding quality
 * @param {string} text - Raw text
 * @returns {string} Cleaned text
 */
function cleanText(text) {
  return text
    .replace(/\s+/g, ' ')           // Replace multiple spaces with a single space
    .replace(/[\r\n]+/g, ' ')       // Replace newlines with spaces
    .replace(/[^\w\s.,?!;:()[\]{}]/g, '')  // Remove special characters except punctuation
    .trim();
}

/**
 * ANN-based search & display with improved accuracy
 */
async function searchAndDisplayResults(annIndex, query, topK, texts, metadata) {
  // Setup progress bar
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(1, 0); // Just one task - searching
  
  // Pre-process query the same way as documents
  const cleanedQuery = cleanText(query);
  
  // Embed the query and ensure normalization
  const qvec = normalizeVector((await createEmbeddingsLocal([cleanedQuery]))[0]);
  
  // k-NN search - remove the third parameter that's causing the error
  const { neighbors, distances } = annIndex.searchKnn(qvec, topK * 2);
  
  progressBar.update(1);
  progressBar.stop();
  console.log();
  
  // When using cosine space, the distances are already in [0,2] range
  // Convert to similarity scores in [0,1] range where 1 is most similar
  const scores = distances.map(d => 1 - (d / 2));
  
  // Filter results with low scores
  const minScoreThreshold = 0.5; // Only show reasonably good matches
  
  // Create result items
  let results = neighbors.map((idx, i) => ({
    index: idx,
    score: scores[i],
    meta: metadata[idx],
    text: texts[idx]
  }))
  .filter(item => item.score >= minScoreThreshold) // Filter low scores
  .slice(0, topK); // Take top K after filtering
  
  // Display results
  if (results.length === 0) {
    console.log("No matching results found. Try a different query.");
  } else {
    results.forEach((result, rank) => {
      const { meta, score, text } = result;
      console.log(`[${meta.sermon_title}] (${meta.sermon_date}) | ¶${meta.paragraph_id} ${meta.block_idx >= 0 ? `blk ${meta.block_idx}` : '(full paragraph)'}`);
      console.log(`Score: ${score.toFixed(4)} (${Math.round(score * 100)}% match)`);
      console.log(`Text: ${text}\n`);
    });
  }
}

/**
 * Check if a file exists
 * @param {string} filePath - Path to the file
 * @returns {Promise<boolean>} True if file exists, false otherwise
 */
async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

// Run the application
main().catch(console.error);