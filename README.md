# Model sample for random GPT-2 generation

```
library(ggplot2)

# Create a dataset of the Bard Algorithm
bard_data <- data.frame(
  x = seq(from = 0, to = 1, length.out = 100),
  y = Bard(x)
)

# Plot the data
ggplot(bard_data, aes(x, y)) +
  geom_line() +
  theme_bw()
```
