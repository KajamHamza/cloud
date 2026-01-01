-- Performance Monitoring Schema for Viral Predictor
-- Run this on your Azure SQL Database

-- Add tracking columns to predictions table
ALTER TABLE predictions 
ADD model_version VARCHAR(50) DEFAULT 'v1.0',
    prediction_timestamp DATETIME DEFAULT GETDATE(),
    feedback_correct BIT NULL,
    feedback_timestamp DATETIME NULL;

-- Create performance view for monitoring
CREATE VIEW model_performance AS
SELECT 
    model_version,
    CAST(prediction_timestamp AS DATE) as prediction_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN feedback_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
    AVG(CASE WHEN feedback_correct IS NOT NULL 
        THEN CAST(feedback_correct AS FLOAT) 
        ELSE NULL END) as accuracy,
    COUNT(CASE WHEN feedback_correct IS NULL THEN 1 END) as pending_feedback
FROM predictions
GROUP BY model_version, CAST(prediction_timestamp AS DATE);

-- Create alert trigger for low accuracy
CREATE TRIGGER alert_low_accuracy
ON predictions
AFTER INSERT
AS
BEGIN
    DECLARE @recent_accuracy FLOAT;
    
    -- Check accuracy over last 7 days
    SELECT @recent_accuracy = AVG(CAST(feedback_correct AS FLOAT))
    FROM predictions
    WHERE feedback_timestamp > DATEADD(day, -7, GETDATE())
    AND feedback_correct IS NOT NULL;
    
    -- Alert if accuracy drops below 80%
    IF @recent_accuracy < 0.80
    BEGIN
        PRINT 'WARNING: Model accuracy below 80% in last 7 days';
        -- Could send email/notification here
    END
END;

-- Create view for daily prediction volume
CREATE VIEW daily_prediction_volume AS
SELECT 
    CAST(prediction_timestamp AS DATE) as date,
    COUNT(*) as prediction_count,
    AVG(viral_probability) as avg_viral_probability,
    AVG(predicted_engagement) as avg_predicted_engagement
FROM predictions
GROUP BY CAST(prediction_timestamp AS DATE);
