# Video Recognition Models

## Pre-trained Models

The following table provides the pre-trained models.
The *Datasets* column shows the abbreviated names of the datasets used for pre-training.
These abbreviations are as follows: K700 – Kinetics-700, MiT – Moments in Time, SSv2 – Something-Something-v2.

### ViT-Tiny

| Datasets               | Link                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| K700                   | [Download](https://drive.google.com/file/d/13MmIXwFzkTcNEbvEljP5hqVVx8o2yTdd/view?usp=drive_link) |
| K700 + MiT             | [Download](https://drive.google.com/file/d/1YR8ztUQ5GF8bJ9CCeg5ljZp09GO5aE7V/view?usp=drive_link) |
| K700 + MiT + SSv2      | [Download](https://drive.google.com/file/d/1g1P1BNQ-NVFUoZCttgpWxC2A5UBd1UNu/view?usp=sharing)    |
|                        |                                                                                                   |
| VG + K700 + MiT + SSv2 | [Download](https://drive.google.com/file/d/1sTMg3m2s05BB9Mqn54aLZJkm_oKq8pYS/view?usp=drive_link) |

The models are pre-trained on inputs of 16 frames × 112 × 112 pixels.

### ViT-Base

| Datasets               | Link                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| VG + K700 + MiT + SSv2 | [Download](https://drive.google.com/file/d/1QtVFQHTetQXP0uOrS-XBSTRrP_JaW4dA/view?usp=drive_link) |

The models are pre-trained on inputs of 32 frames × 158 × 158 pixels.

## Performance on BEAR benchmark

The pre-trained models are evaluated on [BEAR benchmark](https://github.com/AndongDeng/BEAR), which is a collection of 18 video datasets grouped into 5 domains, to demonstrate their effectiveness in a variety of domains.

| Datasets        | ViT-Tiny |              |                   |                      | ViT-Base             |
| --------------- | -------- | ------------ | ----------------- | -------------------- | -------------------- |
|                 | K700     | K700+<br>MiT | K700+MiT+<br>SSv2 | VG+K700+<br>MiT+SSv2 | VG+K700+<br>MiT+SSv2 |
| XD-Violence     | 86.6     | 86.6         | 87.7              | 87.3                 | 90.7                 |
| UCF-Crime       | 32.6     | 35.4         | 39.6              | 36.1                 | 38.2                 |
| MUVIM           | 100.0    | 100.0        | 100.0             | 99.6                 | 100.0                |
|                 |          |              |                   |                      |                      |
| WLASL           | 46.7     | 41.2         | 48.2              | 59.3                 | 53.8                 |
| Jester          | 96.0     | 96.5         | 96.5              | 96.5                 | 96.1                 |
| UAV-Human       | 23.3     | 24.4         | 25.7              | 27.7                 | 38.4                 |
|                 |          |              |                   |                      |                      |
| CharadesEGO     | 5.4      | 5.6          | 5.8               | 5.9                  | 8.1                  |
| ToyotaSmarthome | 77.7     | 78.6         | 79.3              | 79.3                 | 81.2                 |
| Mini-HACS       | 68.3     | 70.7         | 70.6              | 75.9                 | 82.9                 |
| MPII Cooking    | 46.4     | 45.1         | 48.9              | 49.9                 | 54.5                 |
|                 |          |              |                   |                      |                      |
| Mini-Sports1M   | 47.1     | 48.1         | 48.5              | 50.2                 | 54.9                 |
| FineGym         | 66.9     | 69.1         | 70.1              | 71.6                 | 79.0                 |
| MOD20           | 91.1     | 90.1         | 91.8              | 93.4                 | 94.6                 |
|                 |          |              |                   |                      |                      |
| COIN            | 66.8     | 66.4         | 65.9              | 69.6                 | 77.9                 |
| MECCANO         | 41.4     | 40.3         | 41.0              | 42.3                 | 42.7                 |
| InHARD          | 86.4     | 86.0         | 86.5              | 87.5                 | 88.2                 |
| PETRAW          | 96.5     | 97.0         | 97.0              | 97.3                 | 97.7                 |
| MISAW           | 70.8     | 73.9         | 76.3              | 73.4                 | 77.8                 |
|                 |          |              |                   |                      |                      |
| Macro Avg.      | 63.9     | 64.2         | 65.5              | 66.8                 | 69.8                 |