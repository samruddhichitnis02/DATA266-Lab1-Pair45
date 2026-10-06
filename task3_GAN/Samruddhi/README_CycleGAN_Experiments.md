# CycleGAN Monet–Photo Image-to-Image Translation

## Project Overview

This project implements an unpaired CycleGAN for translating ordinary photographs into Monet-style paintings and translating Monet paintings back into photographs.

The project was developed as an image-generation and deep-learning experimentation study. The main objective was to improve the competition score by comparing training schedules, checkpoint-selection strategies, generator upsampling, discriminator design, and fine-tuning settings.

The implementation uses only a CycleGAN trained on the project dataset. No pretrained image-generation model was used to generate the submitted images.

## Problem Formulation

The two image domains are unpaired:

- Domain A: ordinary photographs
- Domain B: Monet paintings

The model learns two mappings:

$$
G_{AB}: A \rightarrow B
$$

$$
G_{BA}: B \rightarrow A
$$

Because the domains are unpaired, the model is trained with adversarial and cycle-consistency objectives rather than pixel-level paired supervision.

## Dataset Split

The dataset was split independently for the Monet and photograph domains.

| Domain | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Monet paintings | 240 | 30 | 30 | 300 |
| Photographs | 5,630 | 704 | 704 | 7,038 |

The training data was used for unpaired CycleGAN optimization. Validation data was used for monitoring experiments, while the test images were reserved for final prediction generation and submission.

## Final Architecture

The best-performing configuration used the original CycleGAN-style architecture rather than the later larger-receptive-field discriminator variant.

### Generators

Two generators were trained:

- `G_AB`: photograph → Monet
- `G_BA`: Monet → photograph

Each generator used the following 256×256 architecture:

| Stage | Layers | Purpose |
|---|---|---|
| Input stem | Reflection padding → 7×7 convolution → InstanceNorm → ReLU | Preserve boundaries and extract low-level features |
| Downsampling 1 | 3×3 convolution, stride 2, 64→128 channels | Reduce spatial resolution |
| Downsampling 2 | 3×3 convolution, stride 2, 128→256 channels | Build compact feature representations |
| Transformation | 9 residual blocks | Learn style and domain-specific changes |
| Upsampling 1 | Transposed convolution, 256→128 channels | Restore spatial resolution |
| Upsampling 2 | Transposed convolution, 128→64 channels | Restore the original image size |
| Output head | Reflection padding → 7×7 convolution → Tanh | Produce a three-channel image in `[-1, 1]` |

The nine residual blocks were selected because they are the standard CycleGAN capacity for 256×256 images. Instance normalization was used to reduce instance-specific contrast and lighting variation, which is useful for artistic style transfer.

### Discriminators

Each domain had its own PatchGAN discriminator:

- `D_A`: distinguishes real and generated photographs
- `D_B`: distinguishes real and generated Monet paintings

The discriminator used a 70×70 PatchGAN design with convolutional feature widths of 64, 128, 256, and 512. Rather than producing one score for the whole image, it classified local image patches. This encouraged realistic texture and brush-stroke structure.

## Training Objective

The generator objective combined three components:

$$
\mathcal{L}_G =
\mathcal{L}_{GAN}
+ \lambda_{cycle}\mathcal{L}_{cycle}
+ \lambda_{identity}\mathcal{L}_{identity}
$$

### Adversarial Loss

Least-squares adversarial loss was used to make translated images appear to belong to the target domain. This is generally more stable for image translation than a binary cross-entropy GAN loss.

### Cycle-Consistency Loss

The cycle constraint required the translated image to preserve the original content:

$$
G_{BA}(G_{AB}(A)) \approx A
$$

$$
G_{AB}(G_{BA}(B)) \approx B
$$

An L1 cycle-consistency loss was used with `lambda_cycle = 10`.

### Identity Loss

Identity loss encouraged a generator to preserve an image that was already in its target domain. The baseline and v3 runs used `lambda_identity = 5`. The fine-tuning run reduced this to `2` to allow stronger style changes, especially in the Photo→Monet direction.

### Other Training Settings

| Setting | Value |
|---|---:|
| Image size | 256×256 |
| Optimizer | Adam |
| Adam β₁, β₂ | 0.5, 0.999 |
| Initial learning rate | 0.0002 |
| Cycle weight | 10 |
| Identity weight | 5 in baseline/v3; 2 in fine-tuning |
| Replay buffer | 50 generated images |
| Input normalization | `[-1, 1]` |
| Output activation | Tanh |
| Random seed in v3/v4 | 42 |

## Experiment History

The competition score is negative, so a score closer to zero is better. The submission files contain FID and MiFID. For consistent comparison, the equivalent score reported below is:

$$
\text{Equivalent score}
=
-\frac{FID + MiFID}{2}
$$

### Experiment 0 — Teaching Assistant Reference Baseline

This was the reference submission supplied for comparison.

| Metric | Result |
|---|---:|
| FID | 156.0362 |
| MiFID | 0.4192 |
| Equivalent score | −78.2277 |

This established the starting point. The relatively high FID indicated a large distribution mismatch between generated and real images.

### Experiment 1 — Standard CycleGAN with Early Stopping

This run used the original 9-residual-block generator and standard 70×70 PatchGAN discriminator. It trained for up to 100 epochs with early stopping. Training stopped after the monitored score failed to improve, while the best checkpoint was from approximately epoch 53.

| Configuration | Value |
|---|---|
| Generator | Standard 9-residual-block CycleGAN generator |
| Discriminator | Standard 70×70 PatchGAN |
| Batch size | 10 |
| Maximum epochs | 100 |
| Cycle weight | 10 |
| Identity weight | 5 |
| Early stopping | Enabled |

| Metric | Result |
|---|---:|
| FID | 100.1885 |
| MiFID | 0.2222 |
| Equivalent score | **−50.2053** |

This was the best overall submission. It reduced FID substantially relative to the reference baseline and produced the strongest MiFID among the recorded submissions.

### Experiment 2 — Bilinear Upsampling and Spectral-Normalized Discriminator

The second experiment changed the architecture in two ways:

1. Transposed convolutions in the generator were replaced with bilinear upsampling followed by convolution.
2. The discriminator used spectral normalization and a larger approximately 142×142 receptive field.

The purpose was to reduce checkerboard artifacts and give the discriminator more global context.

| Configuration | Value |
|---|---|
| Generator upsampling | Bilinear upsampling + 3×3 convolution |
| Discriminator | Spectral-normalized larger PatchGAN |
| Generator learning rate | 0.0002 |
| Discriminator learning rate | 0.0001 |
| Batch size | 16 |
| Epochs | 100 |
| Decay start | Epoch 50 |
| Replay buffer | 50 |
| Cycle weight | 10 |
| Identity weight | 5 |

| Metric | Result |
|---|---:|
| FID | 120.5135 |
| MiFID | 0.4174 |
| Equivalent score | **−60.4655** |

This experiment did not improve the result. The larger and spectrally normalized discriminator appears to have made the adversarial game more difficult for the generator under this training schedule.

The result demonstrates that a theoretically attractive architectural modification is not automatically better. Generator and discriminator capacity must remain balanced, and the learning-rate schedule must be retuned together with the architecture.

### Experiment 3 — Standard Architecture with Batch Size 1 and 150 Epochs

The third experiment returned to the standard architecture and changed the training procedure. It used batch size 1, a fixed seed, a 50-image replay buffer, linear learning-rate decay, periodic local FID evaluation, and checkpoint saving.

| Configuration | Value |
|---|---|
| Generator/discriminator | Original standard architecture |
| Batch size | 1 |
| Epochs | 150 |
| Learning rate | 0.0002 for generators and discriminators |
| Decay start | Epoch 50 |
| Cycle weight | 10 |
| Identity weight | 5 |
| Replay buffer | 50 |
| Checkpoint selection | Periodic local FID monitoring |
| Seed | 42 |

| Metric | Result |
|---|---:|
| FID | 101.9429 |
| MiFID | 0.4115 |
| Equivalent score | **−51.1772** |

This run was much better than the modified architecture experiment and achieved a similar score to the early-stopping baseline. However, it did not beat Experiment 1 because its MiFID was substantially higher.

This showed that longer training alone did not guarantee better test performance.

### Experiment 4 — Fine-Tuning the Epoch-150 Checkpoint

The final experiment resumed from the epoch-150 checkpoint and performed 11 additional fine-tuning epochs, ending at epoch 161. The learning rates and identity weight were changed to make the late-stage update more conservative and to encourage stronger style transfer.

| Configuration | Value |
|---|---|
| Starting checkpoint | Epoch 150 |
| Fine-tuning epochs | 151–161 |
| Batch size | 1 |
| Generator learning rate | 0.0001 |
| Photo discriminator learning rate | 0.0001 |
| Monet discriminator learning rate | 0.00003 |
| Cycle weight | 10 |
| Identity weight | 2 |
| Decay start | Epoch 150 |
| Seed | 42 |

The best internal fine-tuning point was around epoch 154, but it still did not outperform the earlier standard-architecture submission. The final epoch-161 submission produced:

| Metric | Result |
|---|---:|
| FID | 107.0255 |
| MiFID | 0.4090 |
| Equivalent score | **−53.7173** |

Fine-tuning improved MiFID slightly relative to Experiment 3, but FID became worse. Because FID dominates the combined score, the final result declined.

The experiment therefore showed that late fine-tuning from a converged checkpoint was not sufficient to move the model into a better solution.

## Comparison of All Experiments

| Experiment | Main Change | FID ↓ | MiFID ↓ | Equivalent Score ↑ |
|---|---|---:|---:|---:|
| TA/reference baseline | Provided reference model | 156.0362 | 0.4192 | −78.2277 |
| 1. Standard + early stopping | Original CycleGAN; best checkpoint around epoch 53 | **100.1885** | **0.2222** | **−50.2053** |
| 2. Modified architecture | Bilinear upsampling, spectral normalization, larger PatchGAN | 120.5135 | 0.4174 | −60.4655 |
| 3. Standard, 150 epochs | Batch 1, replay buffer, decay, periodic FID checks | 101.9429 | 0.4115 | −51.1772 |
| 4. Fine-tuning | Epoch-150 resume, lower learning rates, identity weight 2 | 107.0255 | 0.4090 | −53.7173 |

## What Worked

### 1. The Standard CycleGAN Architecture Was the Strongest Choice

The standard generator and 70×70 PatchGAN provided the best balance between image-level structure and local texture.

The modified larger discriminator did not improve the score despite adding spectral normalization.

### 2. Early Stopping Preserved a Strong Checkpoint

The best model was not the final epoch. Selecting the checkpoint around epoch 53 avoided later degradation and produced the best submission.

### 3. Replay-Buffer Training and Learning-Rate Decay Were Useful

The v3 training recipe incorporated common CycleGAN stabilization techniques. These helped produce a competitive result and made the training process more reproducible.

### 4. Reducing Identity Loss Changed the Trade-Off

The fine-tuned run achieved slightly lower MiFID, indicating somewhat better content alignment, but its FID increased.

The generator became more content-preserving without producing a sufficiently realistic target-domain distribution.

## What Did Not Work

### 1. The Larger Spectral-Normalized Discriminator

The second architecture produced a worse FID. The discriminator likely became too strong or too global for the generator and schedule used in that experiment.

The change also altered several interacting components at once, making it difficult to isolate the source of the degradation.

### 2. Training Longer Without Changing the Optimization Problem

The 150-epoch run did not substantially beat the early-stopped model.

Once the model reached a stable equilibrium, additional epochs mainly moved around the same solution rather than discovering a better one.

### 3. Fine-Tuning a Late Checkpoint

Fine-tuning from epoch 150 changed the metrics only modestly. It did not recover the earlier best score because the model had already settled into a suboptimal solution.

A new training run with a different controlled configuration would be more promising than repeatedly extending the same checkpoint.

### 4. Selecting Only by Training Losses

Cycle and identity losses measure reconstruction and preservation, but they do not fully measure whether generated images match the target-domain distribution.

FID and MiFID were more relevant for selecting competition checkpoints.

## Main Technical Interpretation

The results indicate that the primary limitation was generated-image distribution quality, measured mainly by FID.

The best run had both the lowest FID and the lowest MiFID in the recorded submissions.

Later experiments produced similar or slightly better MiFID but failed to lower FID, so their overall scores declined.

This is an important deep-learning lesson: improving one loss component or adding model complexity does not necessarily improve the evaluation metric that matters.

The architecture, discriminator strength, batch size, learning-rate schedule, identity weight, replay buffer, and checkpoint-selection rule must be evaluated as one training system.

## Final Conclusion

The strongest result was obtained with the standard CycleGAN architecture trained with early stopping.

It achieved:

- FID: **100.1885**
- MiFID: **0.2222**
- Equivalent competition score: **−50.2053**

The later architecture modification and fine-tuning experiments were valuable because they tested meaningful hypotheses, but neither improved the final result.

The final project conclusion is:

> For this dataset and training budget, the original CycleGAN design reached the best balance between style transfer, content preservation, and training stability. The main improvement opportunity is not simply increasing the number of epochs or discriminator capacity. It is better FID-guided checkpoint selection and a carefully controlled redesign focused on the weaker Photo→Monet translation direction.

## Project Summary

I implemented an unpaired CycleGAN for Monet–photo translation using two residual generators and two 70×70 PatchGAN discriminators.

I established a reference baseline, then ran controlled experiments involving early stopping, batch-size changes, replay buffers, learning-rate decay, bilinear upsampling, spectral normalization, and late-stage fine-tuning.

The best result came from the standard architecture with early stopping, achieving an FID of 100.19 and MiFID of 0.222.

The modified discriminator architecture and late fine-tuning did not improve the result. This showed that lower reconstruction losses and greater model complexity did not necessarily translate into a lower FID.

I used checkpoint-level evaluation to compare the experiments and concluded that architecture–optimization balance and metric-aligned checkpoint selection were more important than simply training for more epochs.