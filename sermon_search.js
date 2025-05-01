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
  .option('min-results', {
    alias: 'm',
    type: 'number',
    default: 0,
    description: 'Minimum number of results to display, regardless of score'
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
  .option('model', {
    alias: 'm',
    type: 'string',
    default: 'Xenova/bge-large-en-v1.5',
    description: 'Embedding model to use'
  })
  .option('verify-model', {
    alias: 'vm',
    type: 'boolean',
    default: false,
    description: 'Show detailed model information'
  })
  .option('server', {
    alias: 's',
    type: 'boolean',
    default: false,
    description: 'Run in server mode (stay active between queries)'
  })
  .help()
  .argv;

// File paths
const SERMON_FILE = args.dataFile;
const EMBEDDINGS_BINARY_FILE = 'sermon_embeddings.bin';
const METADATA_CACHE_FILE = 'sermon_metadata.bin'; // Changed to binary extension
const TEXTS_CACHE_FILE = 'sermon_texts.bin'; // Changed to binary extension
const INDEX_CACHE_FILE = 'sermon_index.bin';
const EMBEDDINGS_CHECKPOINT_FILE = 'sermon_embeddings_checkpoint.bin';

// Constants
// let EMBEDDING_DIM = 1024; // Updated embedding dimension
let EMBEDDING_DIM = 384; // Updated embedding dimension

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
    
    console.log(`Initializing embedding model: ${args.model}`);
    const startTime = Date.now();
    
    try {
      embedder = await pipeline('feature-extraction', args.model, {
        device,
        revision: 'main',
        quantized: false,
      });
      console.log(`Successfully loaded model ${args.model} using ${device.toUpperCase()}`);
      console.log(`Initialization took ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
      
      if (args.verifyModel) {
        await verifyModel(embedder, args.model);
      }
    } catch (err) {
      console.warn(`Failed to initialize ${args.model} on ${device}, falling back to CPU:`, err.message);
      try {
        embedder = await pipeline('feature-extraction', args.model, {
          device: 'cpu',
          revision: 'main',
          quantized: false,
        });
        console.log(`Successfully loaded model ${args.model} using CPU`);
        if (args.verifyModel) {
          await verifyModel(embedder, args.model);
        }
      } catch (fallbackErr) {
        console.error(`Failed to load model ${args.model} on CPU as well:`, fallbackErr.message);
        console.log(`Falling back to default model Xenova/all-MiniLM-L6-v2`);
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
          device: 'cpu',
          revision: 'main',
          quantized: false,
        });
      }
    }
  } catch (err) {
    console.error('Failed to initialize embedder:', err.message);
    throw err;
  }
}

/**
 * Verify the model by checking its properties and running a test embedding
 */
async function verifyModel(model, modelName) {
  console.log('\n--- MODEL VERIFICATION ---');
  console.log(`Model ID: ${modelName}`);
  
  // Extract model info
  const info = model.processor?.tokenizer?.model_max_length ? 
               `Max sequence length: ${model.processor.tokenizer.model_max_length}` : 
               'Model info not available';
  
  console.log(info);
  
  // Test embedding dimensions
  console.log('Testing embedding dimensions...');
  const testText = "This is a test sentence to verify embedding dimensions.";
  const testEmbedding = await model(testText, { pooling: 'mean', normalize: true });
  
  let dimensions = 0;
  if (testEmbedding.data) {
    dimensions = testEmbedding.data.length;
  } else if (Array.isArray(testEmbedding)) {
    dimensions = Array.isArray(testEmbedding[0]) ? 
                testEmbedding[0].length : 
                testEmbedding.length;
  }
  
  console.log(`Embedding dimensions: ${dimensions}`);
  
  // Check if the embedding dimensions match what we expect
  if (dimensions !== EMBEDDING_DIM) {
    console.warn(`⚠️ WARNING: Model produces embeddings with dimension ${dimensions}, but code expects ${EMBEDDING_DIM}`);
    console.warn(`You should update the EMBEDDING_DIM constant to match!`);
  } else {
    console.log(`✓ Embedding dimensions match expected value (${EMBEDDING_DIM})`);
  }
  console.log('---------------------------\n');
  
  // If dimensions don't match, update the constant
  if (dimensions !== EMBEDDING_DIM) {
    console.log(`Updating EMBEDDING_DIM from ${EMBEDDING_DIM} to ${dimensions}`);
    EMBEDDING_DIM = dimensions;
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
 * Save embedding checkpoint to resume from interruptions
 * @param {Array<Array<number>>} embeddings - Embeddings created so far
 * @param {number} lastProcessedIndex - Index of last processed text
 */
async function saveEmbeddingCheckpoint(embeddings, lastProcessedIndex) {
  // Calculate buffer size: header (12 bytes) + embeddings data
  const bufferSize = 12 + (embeddings.length * EMBEDDING_DIM * 4);
  const buffer = Buffer.alloc(bufferSize);
  
  // Write header: embedding dimension, count, and last processed index
  buffer.writeUInt32LE(EMBEDDING_DIM, 0);
  buffer.writeUInt32LE(embeddings.length, 4);
  buffer.writeUInt32LE(lastProcessedIndex, 8);
  
  // Write embedding data
  let offset = 12;
  for (const embedding of embeddings) {
    for (const value of embedding) {
      buffer.writeFloatLE(value, offset);
      offset += 4;
    }
  }
  
  // Write to checkpoint file
  await fs.writeFile(EMBEDDINGS_CHECKPOINT_FILE, buffer);
}

/**
 * Load embedding checkpoint to resume processing
 * @returns {Object|null} Checkpoint data or null if no checkpoint exists
 */
async function loadEmbeddingCheckpoint() {
  try {
    // Check if checkpoint file exists
    if (!await fileExists(EMBEDDINGS_CHECKPOINT_FILE)) {
      return null;
    }
    
    // Read the checkpoint file
    const buffer = await fs.readFile(EMBEDDINGS_CHECKPOINT_FILE);
    
    // Read header information
    const dimension = buffer.readUInt32LE(0);
    const count = buffer.readUInt32LE(4);
    const lastProcessedIndex = buffer.readUInt32LE(8);
    
    // Validate dimension
    if (dimension !== EMBEDDING_DIM) {
      console.warn(`Warning: Checkpoint has different dimension (${dimension}) than current setting (${EMBEDDING_DIM}). Ignoring checkpoint.`);
      return null;
    }
    
    // Read embeddings
    const embeddings = [];
    let offset = 12;
    
    for (let i = 0; i < count; i++) {
      const embedding = new Array(dimension);
      for (let j = 0; j < dimension; j++) {
        embedding[j] = buffer.readFloatLE(offset);
        offset += 4;
      }
      embeddings.push(embedding);
    }
    
    console.log(`Loaded checkpoint with ${embeddings.length} embeddings. Resuming from index ${lastProcessedIndex + 1}`);
    return { embeddings, lastProcessedIndex };
    
  } catch (error) {
    console.warn(`Error loading checkpoint: ${error.message}. Starting from beginning.`);
    return null;
  }
}

/**
 * Create embeddings locally via Xenova with progress indication and checkpointing
 * @param {Array<string>} texts 
 * @returns {Promise<Array<Array<number>>>}
 */
async function createEmbeddingsLocal(texts) {
  if (!embedder) {
    await initEmbedder();
  }
  
  // Check for existing checkpoint
  let all = [];
  let startIndex = 0;
  
  const checkpoint = await loadEmbeddingCheckpoint();
  if (checkpoint) {
    all = checkpoint.embeddings;
    startIndex = checkpoint.lastProcessedIndex + 1;
    
    // Validate checkpoint against current texts
    if (startIndex > texts.length) {
      console.warn(`Warning: Checkpoint index (${startIndex}) exceeds text count (${texts.length}). Starting from beginning.`);
      all = [];
      startIndex = 0;
    }
  }
  
  // Create progress bar
  const progressBar = new cliProgress.SingleBar({
    format: 'Creating embeddings [{bar}] {percentage}% | {value}/{total} texts | ETA: {eta}s',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
  }, cliProgress.Presets.shades_classic);
  
  progressBar.start(texts.length, startIndex);
  
  // Save checkpoint every N items (avoid too frequent disk writes)
  const CHECKPOINT_INTERVAL = 10;
  
  try {
    for (let i = startIndex; i < texts.length; i++) {
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
      
      // Save checkpoint at regular intervals
      if ((i + 1) % CHECKPOINT_INTERVAL === 0 || i === texts.length - 1) {
        await saveEmbeddingCheckpoint(all, i);
        if (args.debug) {
          console.log(`Saved checkpoint at index ${i}`);
        }
      }
    }
  } catch (error) {
    console.error(`Error during embedding creation: ${error.message}`);
    console.log(`Progress has been saved to checkpoint. Restart the process to continue.`);
    process.exit(1); // Exit with error code
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
  const normalizedEmbeddings = validEmbeddings.map(normalizeVector);
  
  // Delete checkpoint file after successful completion
  try {
    await fs.unlink(EMBEDDINGS_CHECKPOINT_FILE);
    if (args.debug) {
      console.log("Checkpoint file removed after successful completion");
    }
  } catch (error) {
    // Ignore errors if file doesn't exist
    if (args.debug) {
      console.log(`Note: Could not remove checkpoint file: ${error.message}`);
    }
  }
  
  return normalizedEmbeddings;
}

/**
 * Save embeddings in binary format (much more efficient than JSON)
 * @param {Array<Array<number>>} embeddings - The embeddings to save
 * @param {string} filename - File to save to
 */
async function saveBinaryEmbeddings(embeddings, filename) {
  const buffer = Buffer.alloc(4 + (4 + embeddings.length * EMBEDDING_DIM * 4));
  
  // Write header: embedding dimension and count
  buffer.writeUInt32LE(EMBEDDING_DIM, 0);
  buffer.writeUInt32LE(embeddings.length, 4);
  
  let offset = 8;
  for (const embedding of embeddings) {
    for (const value of embedding) {
      buffer.writeFloatLE(value, offset);
      offset += 4;
    }
  }
  
  await fs.writeFile(filename, buffer);
}

/**
 * Load embeddings from binary format
 * @param {string} filename - File to load from
 * @returns {Array<Array<number>>} The loaded embeddings
 */
async function loadBinaryEmbeddings(filename) {
  const buffer = await fs.readFile(filename);
  
  const dimension = buffer.readUInt32LE(0);
  const count = buffer.readUInt32LE(4);
  
  const embeddings = [];
  let offset = 8;
  
  for (let i = 0; i < count; i++) {
    const embedding = new Array(dimension);
    for (let j = 0; j < dimension; j++) {
      embedding[j] = buffer.readFloatLE(offset);
      offset += 4;
    }
    embeddings.push(embedding);
  }
  
  return embeddings;
}

/**
 * Save metadata in binary format
 * @param {Array<TextMetadata>} metadata - The metadata to save
 * @param {string} filename - File to save to
 */
async function saveBinaryMetadata(metadata, filename) {
  // Convert metadata to JSON string
  const jsonData = JSON.stringify(metadata);
  // Convert JSON string to buffer
  const buffer = Buffer.from(jsonData);
  // Write buffer to file
  await fs.writeFile(filename, buffer);
}

/**
 * Load metadata from binary format
 * @param {string} filename - File to load from
 * @returns {Promise<Array<TextMetadata>>} The loaded metadata
 */
async function loadBinaryMetadata(filename) {
  const buffer = await fs.readFile(filename);
  const jsonData = buffer.toString();
  return JSON.parse(jsonData);
}

/**
 * Save texts in binary format
 * @param {Array<string>} texts - The texts to save
 * @param {string} filename - File to save to
 */
async function saveBinaryTexts(texts, filename) {
  // Calculate total buffer size needed
  let totalSize = 4; // Space for texts count (uint32)
  const textBuffers = texts.map(text => Buffer.from(text, 'utf8'));
  
  // Calculate total size
  for (let i = 0; i < textBuffers.length; i++) {
    totalSize += 4 + textBuffers[i].length; // 4 bytes for length + text bytes
  }
  
  // Allocate buffer
  const buffer = Buffer.alloc(totalSize);
  
  // Write texts count
  buffer.writeUInt32LE(texts.length, 0);
  
  // Write each text with its length prefix
  let offset = 4;
  for (let i = 0; i < textBuffers.length; i++) {
    buffer.writeUInt32LE(textBuffers[i].length, offset);
    offset += 4;
    textBuffers[i].copy(buffer, offset);
    offset += textBuffers[i].length;
  }
  
  // Write buffer to file
  await fs.writeFile(filename, buffer);
}

/**
 * Load texts from binary format
 * @param {string} filename - File to load from
 * @returns {Promise<Array<string>>} The loaded texts
 */
async function loadBinaryTexts(filename) {
  const buffer = await fs.readFile(filename);
  
  // Read texts count
  const count = buffer.readUInt32LE(0);
  const texts = new Array(count);
  
  // Read each text
  let offset = 4;
  for (let i = 0; i < count; i++) {
    const textLength = buffer.readUInt32LE(offset);
    offset += 4;
    texts[i] = buffer.toString('utf8', offset, offset + textLength);
    offset += textLength;
  }
  
  return texts;
}

/**
 * Calculate lexical match score between query and text
 * @param {string} query - Search query
 * @param {string} text - Text to match against
 * @returns {number} Score between 0-1
 */
function calculateLexicalScore(query, text) {
  // Case-insensitive search for exact matches
  const normalizedQuery = query.toLowerCase();
  const normalizedText = text.toLowerCase();
  
  // Exact match gets highest score
  if (normalizedText.includes(normalizedQuery)) {
    return 1.0;
  }
  
  // For partial matches, calculate word overlap
  const queryWords = normalizedQuery.split(/\s+/);
  const textWords = new Set(normalizedText.split(/\s+/));
  
  // Count matching words
  let matches = 0;
  for (const word of queryWords) {
    if (textWords.has(word)) matches++;
  }
  
  return matches / queryWords.length;
}

/**
 * Save HNSW index to disk
 * @param {HierarchicalNSW} index - The index to save
 * @param {string} filename - File to save to
 */
async function saveIndex(index, filename) {
  console.log(`Saving index to ${filename}...`);
  await index.writeIndex(filename);
  console.log(`Index saved successfully`);
}

/**
 * Load HNSW index from disk
 * @param {string} filename - File to load from
 * @returns {HierarchicalNSW} The loaded index
 */
async function loadIndex(filename) {
  if (!await fileExists(filename)) {
    throw new Error(`Index file ${filename} doesn't exist`);
  }
  
  console.log(`Loading index from ${filename}...`);
  const index = new HierarchicalNSW('cosine', EMBEDDING_DIM);
  await index.readIndex(filename);
  console.log(`Loaded index with ${index.getCurrentCount()} vectors`);
  return index;
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
      !await fileExists(EMBEDDINGS_BINARY_FILE);
      
    if (args.debug) {
      console.log("Debug mode enabled");
      console.log("Using local embeddings via Xenova");
      console.log("Cache files exist check:");
      console.log(`  metadata_cache: ${await fileExists(METADATA_CACHE_FILE)}`);
      console.log(`  texts_cache: ${await fileExists(TEXTS_CACHE_FILE)}`);
      console.log(`  embeddings_binary: ${await fileExists(EMBEDDINGS_BINARY_FILE)}`);
      console.log(`Rebuild needed: ${rebuildNeeded}`);
    }
    
    let texts, metadata, embeddings;
    let filteredTexts, filteredMetadata;  // Declare these variables at this scope
    
    if (!rebuildNeeded) {
      console.log("Loading cached data...");
      
      // Load texts and metadata from binary files
      console.log("Loading binary metadata from disk...");
      metadata = await loadBinaryMetadata(METADATA_CACHE_FILE);
      
      console.log("Loading binary texts from disk...");
      texts = await loadBinaryTexts(TEXTS_CACHE_FILE);
      
      // Load embeddings from binary file
      console.log("Loading binary embeddings from disk...");
      embeddings = await loadBinaryEmbeddings(EMBEDDINGS_BINARY_FILE);
      
      console.log(`Loaded ${texts.length} cached text blocks with embeddings`);
      
      // Set filtered texts and metadata to the loaded values
      filteredTexts = texts;
      filteredMetadata = metadata;
    } else {
      console.log("Processing sermon data...");
      
      // Load sermons from file with filtering options
      const sermons = await loadSermonData(SERMON_FILE, {
        limit: args.limit,
        direction: args.direction
      });
      console.log(`Processed ${sermons.length} sermons from ${SERMON_FILE}`);
      
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
      
      // Save data to cache files - all in binary format
      console.log("Saving data to binary cache files...");
      await saveBinaryTexts(texts, TEXTS_CACHE_FILE);
      await saveBinaryMetadata(metadata, METADATA_CACHE_FILE);
      await saveBinaryEmbeddings(embeddings, EMBEDDINGS_BINARY_FILE);
    }

    // Build or load ANN index using hnswlib-node
    let annIndex;
    const indexExists = await fileExists(INDEX_CACHE_FILE);

    if (!rebuildNeeded && indexExists) {
      console.log("Loading cached ANN index...");
      annIndex = await loadIndex(INDEX_CACHE_FILE);
    } else {
      console.log("Building ANN index using hnswlib-node...");

      // Filter embeddings as before
      const validIndices = [];
      embeddings.forEach((vec, i) => {
        if (Array.isArray(vec) && vec.length === EMBEDDING_DIM) {
          validIndices.push(i);
        } else {
          console.warn(`Skipping embedding #${i}: expected length ${EMBEDDING_DIM}, got ${vec.length}`);
        }
      });
      const filteredEmbeddings = validIndices.map(i => embeddings[i]);
      filteredTexts = validIndices.map(i => texts[i]);
      filteredMetadata = validIndices.map(i => metadata[i]);

      // Build index
      annIndex = new HierarchicalNSW('cosine', EMBEDDING_DIM);
      annIndex.initIndex(filteredEmbeddings.length);
      filteredEmbeddings.forEach((vec, idx) => {
        annIndex.addPoint(vec, idx);
      });
      
      // Save index for future use
      await saveIndex(annIndex, INDEX_CACHE_FILE);
    }

    // Now filteredTexts and filteredMetadata are always defined
    if (args.server) {
      const readline = require('readline');
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });
      
      console.log("\n=== SERVER MODE ACTIVE ===");
      console.log("Enter search queries or type 'exit' to quit");
      
      const promptUser = () => {
        rl.question('> ', async (input) => {
          if (input.toLowerCase() === 'exit') {
            rl.close();
            return;
          }
          
          if (input.trim()) {
            await searchAndDisplayResults(
              annIndex,
              input,
              args.topK,
              filteredTexts,
              filteredMetadata
            );
          }
          
          promptUser();
        });
      };
      
      promptUser();
    } else if (args.query) {
      console.log(`Searching for query: '${args.query}'`);
      await searchAndDisplayResults(
        annIndex,
        args.query,
        args.topK,
        filteredTexts,
        filteredMetadata
      );
    } else {
      console.log("No search query provided. Use --query to search or --server for interactive mode.");
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
  .replace(/\[.*?\]/g, '')        // Remove square brackets and their content
    .replace(/\s+/g, ' ')           // Replace multiple spaces with a single space
    .replace(/[\r\n]+/g, ' ')       // Replace newlines with spaces
    // .replace(/[^\w\s.,?!;:()[\]{}]/g, '')  // Remove special characters except punctuation
    // replace square brackets and remove content inside them
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
  
  // k-NN search - we'll retrieve more candidates for re-ranking
  const { neighbors, distances } = annIndex.searchKnn(qvec, topK * 4);
  
  progressBar.update(1);
  progressBar.stop();
  console.log();
  
  // When using cosine space, the distances are already in [0,2] range
  // Convert to similarity scores in [0,1] range where 1 is most similar
  const semanticScores = distances.map(d => 1 - (d / 2));
  
  // Create result items with hybrid scoring
  let results = neighbors.map((idx, i) => {
    const semanticScore = semanticScores[i];
    const lexicalScore = calculateLexicalScore(query, texts[idx]);
    
    // New advanced hybrid scoring formula that favors semantic when lexical score is low
    const lexicalWeight = 0.3; 
    // const effectiveLexicalWeight = lexicalWeight * Math.pow(lexicalScore, 0.75);
    const effectiveLexicalWeight = lexicalWeight * Math.pow(lexicalScore, 0.75);
    const hybridScore = (semanticScore * (1 - effectiveLexicalWeight)) + (lexicalScore * effectiveLexicalWeight);
    
    return {
      index: idx,
      semanticScore,
      lexicalScore,
      hybridScore,
      meta: metadata[idx],
      text: texts[idx]
    };
  })
  // Filter out empty texts or extremely short texts
  .filter(item => item.text && item.text.trim().length > 5)
  // Then sort by score
  .sort((a, b) => b.hybridScore - a.hybridScore);
  
  // Handle minimum results requirement
  const minResults = Math.min(args.minResults || 0, results.length);
  const filteredResults = minResults > 0 
    ? results.slice(0, Math.max(minResults, results.filter(item => item.hybridScore >= 0.4).length))
    : results.filter(item => item.hybridScore >= 0.4);
    
  // Final results limited by topK
  const finalResults = filteredResults.slice(0, topK);
  
  // Display results
  if (finalResults.length === 0) {
    console.log("No matching results found. Try a different query or lower the threshold.");
  } else {
    finalResults.forEach((result, rank) => {
      const { meta, semanticScore, lexicalScore, hybridScore, text } = result;
      console.log(`[${meta.sermon_title}] (${meta.sermon_date}) | ¶${meta.paragraph_id} ${meta.block_idx >= 0 ? `blk ${meta.block_idx}` : '(full paragraph)'}`);
      console.log(`Score: ${hybridScore.toFixed(4)} (${(hybridScore * 100).toFixed(0)}% match) (Semantic: ${semanticScore.toFixed(2)}, Lexical: ${lexicalScore.toFixed(2)})`);
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