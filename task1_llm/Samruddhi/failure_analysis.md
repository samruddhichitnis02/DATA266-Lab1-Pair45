
# Task 1 : Sequence Model Failure Analysis

## Failure Case 1: Repetition

### Generated snippet

> One day, Timmy wrote a letter to his friend, Timmy. Timmy asked, "What kind of person do you write letters?" Timmy replied...

### Failure type

Repetition

### Observation

The model repeatedly generated the name “Timmy.” This indicates that the model learned the name as a common local pattern but could not maintain enough long-range context to introduce different characters or ideas.

---

## Failure Case 2: Semantically Incorrect Phrase

### Generated snippet

> He loved to write with his jewels.

### Failure type

Semantic inconsistency

### Observation

The sentence is grammatically understandable, but “write with his jewels” is unusual and does not fit the context. The model produced a locally plausible sequence of words but did not fully understand the meaning of the sentence.

---

## Failure Case 3: Loss of Coherence

### Generated snippet

> Timmy asked, "What kind of person do you write letters?" Timmy replied, "I write how fast I can write letters and stories." Timmy

### Failure type

Loss of coherence and incomplete ending

### Observation

The dialogue becomes unnatural and the generated story ends abruptly. The model appears to predict short-range character patterns correctly but struggles to maintain a logical conversation and complete the story.
