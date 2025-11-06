library(ggplot2)
library(data.table)

d <- fread("history.csv")
d
dm <- melt(d, id.vars = c("platform","year","size"))
dm

colnames(dm) <- c("platform","year","size","software","runtime")

ggplot(dm, aes(x = year, y = runtime, color = software)) +
  geom_line(data = subset(dm, year <= 2024 | platform=="CPU")) +
  geom_line(data = subset(dm, year >= 2024 & platform=="GPU"), linetype = "dashed") +
  geom_point() +
  scale_y_log10() +
  facet_grid(size ~ platform) +
  theme_minimal()
