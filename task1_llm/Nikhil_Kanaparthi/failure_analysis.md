# Task 1 Failure Analysis

I reviewed the greedy and temperature-based generations from the final
checkpoint. The examples below are copied from the generated outputs rather
than created independently.

## Case 1 — Repetition

- **Snippet:** “One day, Lily went to the park with her mommy. She saw a big box of colorful colors and she wanted to play with it. She was so excited to see what she could do. She wanted to play with it...”
- **Observation:** The model repeats the same action and phrase instead of advancing the story. This suggests that it has learned common local story patterns but has weak long-range planning.
- **Testable fix:** Compare the repeated 4-gram rate after applying a repetition penalty during sampling. A successful fix should reduce the repeated 4-gram rate without substantially reducing validation accuracy.

## Case 2 — Broken grammar and semantic repetition

- **Snippet:** “Lily saw a big button with a big button.”
- **Observation:** The model identifies a plausible object but repeats the noun phrase and produces an unnatural sentence. This is a local grammar and semantic-redundancy failure.
- **Testable fix:** Train with a longer context length or additional epochs, then compare validation loss and sampled Distinct-2 against the current run.

## Case 3 — Loss of story coherence and unrealistic dialogue

- **Snippet:** “Lily was scared and said, ‘Oh no, Lily, you try to open the button to do something cool. It's time to go home.’”
- **Observation:** The dialogue is not natural because Lily refers to herself by name and the sentence combines conflicting actions. The model loses track of the character's role and the surrounding narrative.
- **Testable fix:** Increase the context length and model capacity, then evaluate whether character references and dialogue structure become more consistent in generated samples.
