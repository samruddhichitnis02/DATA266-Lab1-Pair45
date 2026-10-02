# TinyStories data

The Task 1 implementation downloads TinyStories through the Hugging Face
`datasets` library at runtime. The raw dataset is intentionally not committed
to GitHub because it is large and can be reproduced from the dataset name.

The run uses the `roneneldan/TinyStories` dataset and creates a member-specific
100,000-story training split and 10,000-story validation split using the seed
recorded in each member's configuration.
