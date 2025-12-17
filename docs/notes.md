First implementation of RAG resume generator:

The RAG retrieval will:

Return all resume data initially (full context)
After job analysis, retrieve only the most relevant 10 achievements
Group achievements by job (minimum 2 per job)
Retrieve personality traits matching job soft skills

Key Features Implemented:
✅ Bullet-level chunking - Each achievement is a separate chunk with job context metadata
✅ Minimum 2 achievements per job - Filters out jobs with only 1 relevant achievement
✅ Full → Filtered retrieval - Initial full retrieval, then filtered after job analysis
✅ Personality RAG - Retrieves top 5 personality traits for cover letter enhancement
✅ Backward compatible - Reconstructs YAML structure for existing prompts

Use this portfolio website: https://vercel.com/templates/portfolio/magic-portfolio-for-next-js


## branches
main 
- Prompt version loading all resume data into prompt.

vector_db
- RAG version using job_analysis for semantic search then using prompt to generate response.
- Includes traits ( 16 personalities)

rag_resume
- RAG version using resume_db for semantic search then using prompt to generate response.
- branches from vector_db

hiring_manager
- at some point departed from main. Uses hiring manager parameter that is detected in query and passed into prompt so that it's included in cover letter. Left so that I have a reference for future implementation into RAG version.


What have I learned?

- Chunking strategy depends on the document structrue (resume_ale.md vs personalities_16.md)
- Collections in vector DB can be used to separate different document types for more targeted retrieval or to organize data better.
    - Guiding principle: create separate collections when the data represents fundamentally different semantic spaces, serves different search purposes, has different lifecycle requirements, or needs isolation for security or performance reasons. Think about how your users will search, what they expect to find, and whether mixing the data would produce meaningful or confusing results.
- Metadata vs top-level payload:
    - Promote frequently-filtered values (company, dates, role) to top-level payload fields when you upsert (duplicate is fine). That gives the best server-side filtering and performance. Keep richer context under metadata for display and downstream logic.

In Qdrant you can also filter by metadata fields:
Nested objects
If payload has: {"metadata": {"company": "CFIA"}}
FieldCondition(key="metadata.company", match=MatchValue(value="CFIA"))