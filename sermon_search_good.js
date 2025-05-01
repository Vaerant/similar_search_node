const fs = require('fs/promises');
const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const cliProgress = require('cli-progress');

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
  .help()
  .argv;

// File paths
const SERMON_FILE = args.dataFile;
const EMBEDDINGS_CACHE_FILE = 'sermon_embeddings.json';
const METADATA_CACHE_FILE = 'sermon_metadata.json';
const TEXTS_CACHE_FILE = 'sermon_texts.json';
const INDEX_CACHE_FILE = 'sermon_index.faiss';

// Constants
const EMBEDDING_DIM = 1536; // OpenAI embedding dimension

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
      console.log("Using OpenAI for embeddings");
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
      
      // Load sermons from file
      const sermons = await loadSermonData(SERMON_FILE);
      console.log(`Processed ${sermons.length} sermons from ${SERMON_FILE}`);
      
      // Extract text and metadata from sermons
      const { processedTexts, processedMetadata } = flattenSermonContent(sermons);
      texts = processedTexts;
      metadata = processedMetadata;
      console.log(`Indexed ${texts.length} text blocks from ${sermons.length} sermons`);
      
      // Create embeddings using OpenAI API
      console.log("Creating embeddings using OpenAI API...");
      const startTime = Date.now();
      embeddings = await createEmbeddingsOpenAI(texts);
      console.log(`Created ${embeddings.length} embeddings in ${(Date.now() - startTime) / 1000}s`);
      
      // Save data to cache files
      console.log("Saving data to cache...");
      await fs.writeFile(TEXTS_CACHE_FILE, JSON.stringify(texts));
      await fs.writeFile(METADATA_CACHE_FILE, JSON.stringify(metadata));
      await fs.writeFile(EMBEDDINGS_CACHE_FILE, JSON.stringify(embeddings));
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
    const data = await fs.readFile(filePath, 'utf-8');
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
 * Create embeddings using OpenAI API
 * @param {Array<string>} texts - Array of text strings
 * @returns {Promise<Array<Array<number>>>} Array of embedding vectors
 */
async function createEmbeddingsOpenAI(texts) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set in environment variables or .env file");
  }
  
  const batchSize = 50; // OpenAI recommends smaller batches
  const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
  progressBar.start(Math.ceil(texts.length / batchSize), 0);
  
  const allEmbeddings = [];
  
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    
    // Implement retry with exponential backoff
    let retryCount = 0;
    const maxRetries = 5;
    let waitTime = 30; // Initial wait time in seconds
    
    while (true) {
      try {
        const response = await axios({
          method: 'post',
          url: 'https://api.openai.com/v1/embeddings',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
          },
          data: {
            model: 'text-embedding-3-small',
            input: batch
          }
        });
        
        // Extract embeddings from response
        const batchEmbeddings = response.data.data.map(item => item.embedding);
        allEmbeddings.push(...batchEmbeddings);
        break; // Success, exit retry loop
      } catch (error) {
        // Handle rate limiting
        if (error.response && error.response.status === 429) {
          if (retryCount >= maxRetries) {
            throw new Error(`Rate limit exceeded after ${maxRetries} retries`);
          }
          
          retryCount++;
          console.log(`Rate limit hit on batch ${Math.floor(i/batchSize) + 1}, retry ${retryCount}/${maxRetries}. Waiting for ${waitTime} seconds...`);
          await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
          waitTime *= 2; // Exponential backoff
        } else {
          throw new Error(`API error: ${error.message}`);
        }
      }
    }
    
    // Save progress periodically
    if (allEmbeddings.length % 1000 === 0) {
      const tempFile = `temp_embeddings_${allEmbeddings.length}.json`;
      await fs.writeFile(tempFile, JSON.stringify(allEmbeddings));
      console.log(`Saved progress to ${tempFile}`);
    }
    
    progressBar.update(Math.floor(i / batchSize) + 1);
    
    // Respect rate limits
    const sleepTime = 200 + (Math.floor(i / batchSize) % 10) * 100; // 200-1100ms
    await new Promise(resolve => setTimeout(resolve, sleepTime));
  }
  
  progressBar.stop();
  return allEmbeddings;
}

/**
 * Brute‑force cosine similarity search
 */
async function searchAndDisplayResults(embeddings, query, topK, texts, metadata) {
  // get query vector
  const [qvec] = await createEmbeddingsOpenAI([query]);

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