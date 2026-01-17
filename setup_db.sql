-- DocuMint Database Setup
-- Run this script to manually create the database and tables

-- Step 1: Create database (run as postgres superuser)
-- CREATE DATABASE documint;

-- Step 2: Connect to the database
-- \c documint

-- Step 3: Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 4: Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER,
    total_chunks INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 5: Create document_chunks table with vector column
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,  -- 384 dimensions for all-MiniLM-L6-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 6: Create indexes for performance
-- Index on document_id for fast chunk lookups
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);

-- HNSW index for fast vector similarity search
-- HNSW (Hierarchical Navigable Small World) is faster than IVFFlat for most use cases
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- m = 16: Number of connections per layer (higher = better quality, more memory)
-- ef_construction = 64: Size of dynamic candidate list (higher = better quality, slower build)

-- Step 7: Verify setup
SELECT 
    'Tables created' as status,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('documents', 'document_chunks')) as table_count,
    (SELECT COUNT(*) FROM pg_indexes WHERE indexname LIKE 'idx_chunks%') as index_count;

-- Optional: View table structure
-- \d documents
-- \d document_chunks
