```mermaid
flowchart TD
    A[Prediction] --> B{Is in input?}
    B -- Yes --> E[End]
    B -- No ----> C[Normalization Strategy]
    C --> E[End]
```