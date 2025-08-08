# The Intention-Based Abstraction Layer for AI-Driven Light Show Generation

## 1. The Rationale for Abstraction in Creative AI

The direct application of machine learning models to raw lighting control data, such as DMX-512 streams, presents two fundamental and prohibitive challenges: the **curse of dimensionality** and the **semantic gap**. A single professional light show can generate millions of DMX parameter values per minute, creating a high-dimensional dataset that is computationally intractable for standard deep learning architectures. More critically, these raw values lack inherent artistic meaning; a DMX value of 255 on channel 42 does not, on its own, describe the creative intent of a "sharp, intense strobe" versus a "slow, gentle fade."

To overcome these limitations, this research introduces a crucial abstraction layer. This layer is designed not merely to reduce data, but to translate it into a language of artistic intent. It achieves this through two core principles: **semantic grouping** and **feature extraction**.

### The Principle of Semantic Grouping

First, individual luminaires are clustered into semantically coherent units based on their physical position (e.g., front truss, back truss, stage floor) and aesthetic function (e.g., key light, backlight, effects). This process mirrors the exact workflow of a human lighting designer, who selects a group of fixtures to create a unified look, such as a full-stage red wash. This strategy reduces the output dimensionality from hundreds of individual fixtures to a manageable set of 8-16 semantic groups, allowing the model to learn artistically relevant relationships.

### The Benefits of Abstraction

This approach yields several critical benefits that make AI-driven generation feasible:

* **Achieving a Fixture-Agnostic Model**: By abstracting away the specifics of hardware, the model learns the *what* (the artistic intent, like a rhythmic pulse) rather than the *how* (the specific DMX values for a MAC Aura PXL). This makes the trained model universally applicable. The generated abstract output can be translated back to *any* lighting rig through a separate interpretation step, a critical requirement for real-world usability.

* **Bridging the Semantic Gap**: The layer transforms raw, numerical data into features that represent meaningful artistic concepts like peak intensity, color, and dynamic range. This provides the model with a representation that is directly correlated with creative choices.

* **Enabling Learnability**: This process provides a necessary **inductive bias**, guiding the model toward learning musically and visually coherent patterns. The abstracted features create a cleaner, more stable signal, focusing the learning process on the meaningful relationships between sound and light.

The following sections detail the "Global Intention-Based" component of this layer, designed to capture the flowing, song-length artistic intent of a lighting design.

## 2. The Abstracted Features and Their Formulation

The Global Intention-Based model describes a lighting show as a set of continuous feature curves for each defined group of luminaires. These curves represent the collective behavior and artistic intent for that group over time. The following key parameters are extracted:

* **Intensity of Peak absolute ($I_{\text{peak}}$)**: Represents the maximum brightness of any single fixture within a group at a given moment. This feature captures the absolute peak energy output of the group, essential for identifying accents and moments of high intensity.

  $$
  I_{\text{peak}}(g, t) = \max_{i \in g} \big(I_i(t)\big)
  $$

  where $g$ is the set of luminaires in the group and $I_i(t)$ is the intensity of an individual luminaire.

* **Spatial Intensity Gradient ($\nabla_S I$)**: Measures the sharpness of the intensity distribution *across the fixtures within a group* at a single frame. This feature is crucial for distinguishing between a stark, high-contrast look where a single fixture is highlighted, and a smooth, cohesive wash of light. A high gradient value indicates a sharp falloff in brightness between adjacent fixtures (a "spiky" look), while a low value signifies a smooth, sine-like intensity distribution across the group (a "wash" or "fan"). This is a spatial metric, not a measure of temporal change.

  $$
  \nabla_S I(g, t) = \frac{1}{N_{\text{fixtures}} - 1} \sum_{i=2}^{N_{\text{fixtures}}} \big|I_i(t) - I_{i-1}(t)\big|
  $$

  where fixtures $i$ in group $g$ are spatially ordered, and $I_i(t)$ is the intensity of an individual fixture.

* **AF Peak Density ($\rho_{\text{peak}}$)**: An "Alternating Factor" that quantifies the spatial complexity of the intensity distribution across the group at a single frame. A low value indicates a simple look with one or few dominant light sources, while a high value signifies a complex, "spiky" pattern with many alternating points of light.

  $$
  \rho_{\text{peak}}(g, t) = \frac{N_{\text{peaks}}(g, t)}{N_{\text{fixtures}}}
  $$

  where $N_{\text{peaks}}$ is the number of distinct spatial intensity peaks across the fixtures in group $g$ at time $t$.

* **AF Peak Similarity / Intensity of Minima Inverse ($I_{\text{min\_inv}}$)**: This metric captures the contrast or dynamic range within the group. By inverting the minimum intensity value, a score near 1.0 indicates high contrast (at least one fixture is off), while a score near 0 indicates low contrast (all fixtures are illuminated to some degree).

  $$
  I_{\text{min\_inv}}(g, t) = 1 - \min_{i \in g} \big(I_i(t)\big)
  $$

* **Color Hue Mean ($\bar{H}$)**: The average hue of all fixtures in the group. This represents the dominant color of the group's output, providing a single value for the overall color aesthetic.

  $$
  \bar{H}(g, t) = \text{mean}_{i \in g} \big(H_i(t)\big)
  $$

* **Color Saturation Mean ($\bar{S}$)**: The average saturation of all fixtures in the group. This feature describes the purity or vividness of the dominant color, distinguishing between pastel shades and deeply saturated tones.

  $$
  \bar{S}(g, t) = \text{mean}_{i \in g} \big(S_i(t)\big)
  $$

## 3. The Journey of Data: A Practical Example

To illustrate this transformation, we will examine a 300-frame (10-second) segment from the performance by **Liraz** at the **Roskilde Festival 2023**. We focus on the group of fixtures designated **'AURA PXL LX2'**.

### Step 1: The Raw Data Input

The process begins with the raw, preprocessed DMX data, where each fixture's parameters are available as a time-series array. Figure 1 shows the `BRIGHTNESS` values for three individual fixtures within the 'AURA PXL LX2' group. This representation is noisy, complex, and highly dimensional, making it a poor direct input for a machine learning model.

*Figure 1: The raw brightness values for three individual fixtures from the 'AURA PXL LX2' group. The high-frequency detail and disparate behavior make it difficult for an AI to discern a single, coherent artistic intent.*

### Step 2: The Transformation Process

The raw parameter data for all fixtures in the 'AURA PXL LX2' group is fed into the abstraction algorithms. The multiple streams of brightness, hue, and saturation are aggregated and transformed into a single set of six feature curves for that group, representing its collective behavior according to the formulas in Section 2.

### Step 3: The Abstracted Representation

The output of the abstraction layer is a clean, low-dimensional, and semantically rich representation of the group's behavior. Figure 2 visualizes the six resulting feature curves. This is the final data that the neural network uses for training. The chaotic individual signals have been transformed into a smooth, learnable language of artistic intent.

*Figure 2: The six global intention-based features for the entire 'AURA PXL LX2' group. The raw data has been transformed into a smooth, learnable representation that captures artistic intent, including dynamic plots for intensity-related features and intuitive visualizations for the average group color.*

## 4. Conclusion and Implications

The transformation from the fixture-specific signals in Figure 1 to the coherent, intention-based curves in Figure 2 is the core function of the Intention-Based Abstraction Layer.

By converting raw DMX into a language of intent, this layer allows the neural network to learn the principles of lighting design regarding the brightness distribution and color fixture- and setup-agnostic.
