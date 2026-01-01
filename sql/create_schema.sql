-- FIX THE DATABASE SCHEMA
-- Run this in Azure Portal -> SQL Database -> Query Editor

-- 1. Drop existing table (WARNING: Deletes old data, which is fine for now)
IF OBJECT_ID('dbo.predictions', 'U') IS NOT NULL
DROP TABLE dbo.predictions;
GO

-- 2. Create correct table matching app.py
CREATE TABLE predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    input_text NVARCHAR(MAX),        -- This was missing!
    is_viral_prediction BIT,
    viral_probability FLOAT,
    predicted_engagement INT,
    model_version VARCHAR(50),
    prediction_timestamp DATETIME DEFAULT GETDATE(),
    
    -- Future proofing for feedback loop
    feedback_correct BIT,
    feedback_timestamp DATETIME
);
GO
