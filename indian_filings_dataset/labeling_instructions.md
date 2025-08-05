
# MD&A Sentiment Labeling Instructions

## Overview
This dataset contains Management Discussion and Analysis (MD&A) sections from Indian companies' annual reports. Your task is to label each text segment with its sentiment orientation.

## Sentiment Categories

### 1. Positive
- Optimistic outlook and growth projections
- Strong financial performance highlights
- Successful strategy implementation
- Market expansion achievements
- Innovation and competitive advantages
- Example: "We achieved record revenues this year and expect continued growth in emerging markets."

### 2. Negative
- Declining performance or losses
- Market challenges and threats
- Regulatory difficulties
- Operational problems
- Pessimistic outlook
- Example: "The company faced significant headwinds due to supply chain disruptions and declining demand."

### 3. Neutral
- Factual reporting without emotional tone
- Balanced discussion of opportunities and challenges
- Objective market analysis
- Standard operational updates
- Example: "The company operates in three business segments with diversified revenue streams."

### 4. Mixed
- Contains both positive and negative elements
- Balanced optimism and caution
- Acknowledges challenges while highlighting opportunities
- Example: "While we faced margin pressures in Q1, our new product launches show promising early results."

## Labeling Guidelines

1. **Read the entire text segment** before assigning a label
2. **Consider the overall tone** rather than individual sentences
3. **Focus on management's attitude** toward business prospects
4. **Use context** from financial and operational discussions
5. **Be consistent** across similar text patterns

## Confidence Scoring
Rate your confidence in the label on a scale of 1-5:
- 1: Very uncertain
- 2: Somewhat uncertain  
- 3: Moderately confident
- 4: Confident
- 5: Very confident

## Special Considerations for Indian Companies
- Consider regulatory environment (RBI, SEBI guidelines)
- Account for seasonal business patterns (monsoon, festivals)
- Be aware of government policy impacts
- Consider currency fluctuation discussions
- Note infrastructure and logistics challenges

## Quality Checks
- Ensure labels reflect management sentiment, not just financial numbers
- Check consistency within company across different sections
- Verify mixed labels truly contain both positive and negative elements
- Confirm neutral labels lack emotional indicators

## Notes Section
Use the notes field to capture:
- Reasoning for difficult classifications
- Key phrases that influenced your decision
- Potential ambiguities or edge cases
- Sector-specific considerations
